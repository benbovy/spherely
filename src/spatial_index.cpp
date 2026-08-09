#include <pybind11/stl.h>
#include <s2/encoded_s2shape_index.h>
#include <s2/mutable_s2shape_index.h>
#include <s2/s1angle.h>
#include <s2/s1chord_angle.h>
#include <s2/s2cell_id.h>
#include <s2/s2cell_union.h>
#include <s2/s2closest_edge_query.h>
#include <s2/s2region_coverer.h>
#include <s2/s2shape_index.h>
#include <s2/s2shapeutil_coding.h>
#include <s2/util/coding/coder.h>
#include <s2geography.h>
#include <s2geography/index.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "constants.hpp"
#include "geography.hpp"
#include "predicates.hpp"
#include "pybind11.hpp"
#include "s2_tmp_memory_budget.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

namespace {

// s2 0.11 declares Decoder::get_varint64(uint64*) -- `unsigned long long` -- while
// 0.14 declares get_varint64(uint64_t*), which on LP64 is `unsigned long`. The two
// are distinct types, so a pointer to the wrong one will not convert. Deduce
// whichever this build's s2 wants rather than naming either.
template <typename T>
T varint64_value_type(bool (Decoder::*)(T*));
using VarintU64 = decltype(varint64_value_type(&Decoder::get_varint64));

/*
** Sets s2geometry's process-wide index-build temporary memory budget
** (``FLAGS_s2shape_index_tmp_memory_budget``) for the lifetime of the guard and
** puts the previous value back when it goes out of scope, so an override cannot
** leak out of the build it was requested for.
**
** The flag is read and written through spherely::get/set_s2_tmp_memory_budget,
** which own the declaration it needs (see s2_tmp_memory_budget.cpp) -- reaching
** it directly from this file would not link against a shared s2 on Windows.
*/
class TmpMemoryBudgetGuard {
public:
    explicit TmpMemoryBudgetGuard(std::int64_t bytes) : m_previous(get_s2_tmp_memory_budget()) {
        set_s2_tmp_memory_budget(bytes);
    }

    ~TmpMemoryBudgetGuard() {
        set_s2_tmp_memory_budget(m_previous);
    }

    TmpMemoryBudgetGuard(const TmpMemoryBudgetGuard&) = delete;
    TmpMemoryBudgetGuard& operator=(const TmpMemoryBudgetGuard&) = delete;

private:
    std::int64_t m_previous;
};

}  // namespace

/*
** A spatial index over a collection of Geography objects, backed by
** s2geography::GeographyIndex (a MutableS2ShapeIndex). Provides fast candidate
** lookup similar to shapely's STRtree.
*/
class SpatialIndex {
public:
    SpatialIndex(const py::array_t<PyObjectGeography>& geographies) {
        if (geographies.ndim() != 1) {
            throw py::type_error("geographies must be a 1-dimensional array");
        }

        // store a shallow copy of the input array so that replacing its
        // elements afterwards cannot drop Geography objects whose S2Shapes
        // are still borrowed by the index
        m_geographies = py::array_t<PyObjectGeography>::ensure(geographies.attr("copy")());
        m_index = std::make_unique<s2geog::GeographyIndex>();

        auto n = m_geographies.size();
        auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
        // this is what s2geography::GeographyIndex::Add does, except that it
        // keeps the shape id -> value mapping in a vector it grows with
        // ``reserve(size() + num_shapes)`` per geography. libc++ reserves
        // exactly and the ``resize`` that follows leaves capacity == size, so
        // every call reallocates and copies the whole vector: quadratic, and
        // 15 s of pure vector growth for 555k polygons (0.04 s here).
        // Reserving once and adding the shapes to the shape index directly
        // builds exactly the same index. This is a workaround for the
        // dependency -- it can go away once s2geography grows that vector
        // geometrically.
        auto& mutable_index = m_index->MutableShapeIndex();
        m_values.reserve(static_cast<std::size_t>(n));
        for (py::ssize_t i = 0; i < n; i++) {
            const auto& geog = data[i].as_geog_ptr()->geog();
            for (int k = 0; k < geog.num_shapes(); k++) {
                int shape_id = mutable_index.Add(geog.Shape(k));
                m_values.resize(static_cast<std::size_t>(shape_id) + 1);
                m_values[static_cast<std::size_t>(shape_id)] = static_cast<int>(i);
            }
        }
    }

    // in encoded mode the EncodedS2ShapeIndex points into ``m_encoded``,
    // whose buffer may live inside the string object itself (small string
    // optimization): moving or copying the index would leave it dangling
    SpatialIndex(SpatialIndex&&) = delete;
    SpatialIndex& operator=(SpatialIndex&&) = delete;

    std::size_t size() const {
        if (is_encoded()) {
            return static_cast<std::size_t>(m_num_geographies);
        }
        return static_cast<std::size_t>(m_geographies.size());
    }

    // True once the queued updates have been applied (no build work pending).
    // An index loaded from encoded bytes never has any: its cells are decoded
    // on demand, so there is nothing to build.
    bool is_built() const {
        if (is_encoded()) {
            return true;
        }
        return m_index->ShapeIndex().is_fresh();
    }

    // Apply the updates queued by the constructor, optionally under a raised
    // s2 temporary memory budget. A no-op once the index is built.
    void build(std::optional<std::int64_t> tmp_memory_budget) const {
        if (tmp_memory_budget.has_value() && *tmp_memory_budget <= 0) {
            throw py::value_error("tmp_memory_budget must be a positive number of bytes, got " +
                                  std::to_string(*tmp_memory_budget) +
                                  " (s2geometry's default budget is 104857600 bytes)");
        }

        // an encoded index has no queued updates and no mutable shape index to
        // build: the budget check above is all that applies to it
        if (is_encoded()) {
            return;
        }

        // nothing pending: no build to tune, so leave the process-wide flag alone
        if (is_built()) {
            return;
        }

        const auto& index = m_index->ShapeIndex();
        if (tmp_memory_budget.has_value()) {
            TmpMemoryBudgetGuard guard(*tmp_memory_budget);
            index.ForceBuild();
        } else {
            index.ForceBuild();
        }
    }

    // Number of tree entries whose Geography has been reconstructed from the
    // encoded blob so far (always 0 for a built index). Exposed as the
    // ``_decoded_count`` property so that tests can pin the laziness
    // contract deterministically.
    std::size_t decoded_count() const {
        return m_decoded_count;
    }

    py::array geometries() const {
        check_has_geographies("geometries");
        if (!is_encoded()) {
            return m_geographies;
        }
        // reconstruct every remaining entry (insertion order is the tree
        // order) and cache the resulting array
        if (!m_decoded_geometries) {
            auto result = py::array_t<PyObjectGeography>(m_num_geographies);
            py::buffer_info buf = result.request();
            py::object* data = static_cast<py::object*>(buf.ptr);
            for (py::ssize_t i = 0; i < m_num_geographies; i++) {
                decoded_geog(i);
                data[i] = m_decoded_slots[static_cast<std::size_t>(i)];
            }
            m_decoded_geometries = std::move(result);
        }
        // m_decoded_geometries is stored as a py::object, so borrow a new
        // reference to it as a py::array rather than downcasting the
        // reference itself
        return py::reinterpret_borrow<py::array>(m_decoded_geometries);
    }

    // Serialize the index to bytes: a small header, the shape id -> tree
    // index mapping, then either the Geography objects themselves (a fixed
    // width offset/length table followed by one block per tree entry) or a
    // compact S2 shape stream, and finally the index structure.
    //
    // The shape data is stored once either way. Without the geographies the
    // stream written by CompactEncodeTaggedShapes backs the decoded index
    // directly (via LazyDecodeShapeFactory, which decodes each shape on
    // first access only). With the geographies there is no separate shape
    // stream: the decoded index pulls shape k out of the reconstruction of
    // the geography that owns it (see materialize_shape), which defers all
    // shape data to first touch but makes that touch reconstruct the whole
    // owning geography.
    //
    // Everything past the header is s2geometry's / s2geography's own
    // encoding, so the format is tied to the library versions spherely is
    // built against (only the header is versioned here).
    py::bytes encode(bool include_geographies) const {
        if (is_encoded()) {
            // the original blob is all this index has: hand it back when it
            // matches the request rather than fabricating the other flavor
            if (include_geographies != m_has_geographies) {
                throw py::value_error(
                    m_has_geographies
                        ? "encode(include_geographies=False) is not supported for an index "
                          "loaded from encoded bytes that include the geographies"
                        : "encode(include_geographies=True) is not supported for an index "
                          "loaded from bytes encoded with include_geographies=False");
            }
            return py::bytes(m_encoded);
        }

        // gather the wrapped geographies under the GIL; their S2 data is
        // owned by the Python objects this index keeps alive
        std::vector<const Geography*> geogs;
        if (include_geographies) {
            auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
            geogs.reserve(static_cast<std::size_t>(m_geographies.size()));
            for (py::ssize_t i = 0; i < m_geographies.size(); i++) {
                geogs.push_back(data[i].as_geog_ptr());
            }
        }

        Encoder encoder;

        const auto& index = m_index->ShapeIndex();
        auto num_shapes = index.num_shape_ids();
        auto num_geographies = static_cast<std::uint64_t>(m_geographies.size());

        {
            // encoding is pure C++ work on objects this index keeps alive and
            // can take a while for a large index: let other threads run
            py::gil_scoped_release release;

            // widened to size_t: the header plus one varint per shape value
            // and one for each of the two counts (int would overflow past
            // ~2e8 shapes)
            encoder.Ensure(static_cast<std::size_t>(kHeaderBytes) +
                           static_cast<std::size_t>(Encoder::kVarintMax64) *
                               (static_cast<std::size_t>(num_shapes) + 2));
            encoder.put32(kEncodingMagic);
            encoder.put8(kEncodingVersion);
            encoder.put8(include_geographies ? kFlagHasGeographies : 0);
            encoder.put_varint64(num_geographies);
            encoder.put_varint64(static_cast<std::uint64_t>(num_shapes));
            for (int i = 0; i < num_shapes; i++) {
                encoder.put_varint64(
                    static_cast<std::uint64_t>(m_values.at(static_cast<std::size_t>(i))));
            }

            if (include_geographies) {
                // single copy of the shape data: the geography blocks stand
                // in for the shape stream (see materialize_shape)
                encode_geography_blocks(encoder, geogs);
            } else {
                s2shapeutil::CompactEncodeTaggedShapes(index, &encoder);
            }
            index.Encode(&encoder);
        }
        return py::bytes(encoder.base(), encoder.length());
    }

    // Load an index serialized with ``encode()``, wrapping the bytes in a
    // lazy EncodedS2ShapeIndex: opening is near-instant and index cells /
    // shapes are only decoded when queries touch them.
    static std::unique_ptr<SpatialIndex> from_encoded(const py::buffer& encoded) {
        // any contiguous bytes-like object (bytes, bytearray, memoryview...);
        // the bytes are copied, so the buffer is not kept alive afterwards
        py::buffer_info info = encoded.request();
        if (info.ndim != 1 || info.itemsize != 1 || info.strides[0] != 1) {
            throw py::type_error("encoded must be a contiguous bytes-like object");
        }

        auto self = std::unique_ptr<SpatialIndex>(new SpatialIndex());
        self->m_encoded.assign(static_cast<const char*>(info.ptr),
                               static_cast<std::size_t>(info.size));

        Decoder decoder(self->m_encoded.data(), self->m_encoded.size());
        auto fail = []() {
            throw py::value_error("invalid encoded SpatialIndex");
        };
        if (decoder.avail() < kHeaderBytes || decoder.get32() != kEncodingMagic) {
            fail();
        }
        if (decoder.get8() != kEncodingVersion) {
            throw py::value_error("unsupported encoded SpatialIndex version");
        }
        auto flags = decoder.get8();
        if ((flags & ~kFlagHasGeographies) != 0) {
            throw py::value_error("unsupported encoded SpatialIndex flags");
        }
        self->m_has_geographies = (flags & kFlagHasGeographies) != 0;

        VarintU64 num_geographies;
        VarintU64 num_shapes;
        if (!decoder.get_varint64(&num_geographies) || !decoder.get_varint64(&num_shapes)) {
            fail();
        }
        // both counts come straight from the (untrusted) input: reject values
        // that cannot describe this buffer before allocating anything. Tree
        // indices and shape ids are both stored as int (as in the built index),
        // and every encoded value takes at least one byte.
        constexpr auto kMaxInt = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        if (num_geographies > kMaxInt || num_shapes > kMaxInt || num_shapes > decoder.avail()) {
            fail();
        }
        // a fat blob spends at least one block table entry plus one (minimal)
        // block per geography: a count that cannot fit in what is left of the
        // buffer is rejected here, before the per-geography vectors are sized
        // on it. Without this a 12 byte blob claiming 2^31-1 geographies asks
        // for tens of gigabytes before ``init_geography_blocks`` ever looks at
        // ``avail()``.
        if (self->m_has_geographies &&
            num_geographies > decoder.avail() / (kBlockTableEntryBytes + kBlockMinBytes)) {
            fail();
        }
        self->m_num_geographies = static_cast<py::ssize_t>(num_geographies);
        self->m_values.reserve(num_shapes);
        for (std::uint64_t i = 0; i < num_shapes; i++) {
            VarintU64 value;
            if (!decoder.get_varint64(&value) || value >= num_geographies) {
                fail();
            }
            self->m_values.push_back(static_cast<int>(value));
        }

        if (self->m_has_geographies) {
            self->m_shape_counts.assign(static_cast<std::size_t>(num_geographies), 0);
            self->m_first_shape.assign(static_cast<std::size_t>(num_geographies), -1);
            int previous = -1;
            for (std::size_t shape_id = 0; shape_id < self->m_values.size(); shape_id++) {
                int value = self->m_values[shape_id];
                // materialize_shape addresses shape k as (tree entry, k -
                // first shape of the entry), which requires each entry's
                // shapes to form one contiguous ascending run, as written by
                // encode: a non-monotonic value table cannot come from it
                if (value < previous) {
                    fail();
                }
                if (value > previous) {
                    self->m_first_shape[static_cast<std::size_t>(value)] =
                        static_cast<int>(shape_id);
                }
                self->m_shape_counts[static_cast<std::size_t>(value)]++;
                previous = value;
            }
            self->m_decoded_slots.resize(static_cast<std::size_t>(num_geographies));
            if (!self->init_geography_blocks(decoder, num_geographies)) {
                fail();
            }
        }

        self->m_encoded_index = std::make_unique<EncodedS2ShapeIndex>();
        // a fat blob carries no shape stream of its own: the index pulls
        // shapes out of the geography blocks on demand. A thin blob's shapes
        // are decoded lazily from its compact shape stream.
        bool init_ok =
            self->m_has_geographies
                ? self->m_encoded_index->Init(&decoder, GeographyBlockShapeFactory(self.get()))
                : self->m_encoded_index->Init(&decoder,
                                              s2shapeutil::LazyDecodeShapeFactory(&decoder));
        if (!init_ok) {
            fail();
        }
        // the value table is indexed by shape id: a table that does not cover
        // every decoded shape would be read out of bounds by queries
        if (self->m_values.size() !=
            static_cast<std::size_t>(self->m_encoded_index->num_shape_ids())) {
            fail();
        }
        if (self->m_has_geographies && !self->check_index_shape_ids()) {
            fail();
        }
        return self;
    }

    // Scalar query: single Geography -> 1-d array of sorted tree indices.
    py::array_t<py::ssize_t> query(const Geography& geography,
                                   std::optional<std::string> predicate) const {
        auto pred = make_predicate(predicate);
        auto results = query_one(geography, pred ? &*pred : nullptr);
        return to_index_array(results);
    }

    // Array query: 1-d array of Geography -> (2, K) array of
    // (input index, tree index) pairs.
    py::array_t<py::ssize_t> query(const py::array_t<PyObjectGeography>& geographies,
                                   std::optional<std::string> predicate) const {
        if (geographies.ndim() != 1) {
            throw py::type_error(
                "query geography must be a Geography or a 1-dimensional array of Geography");
        }

        auto pred = make_predicate(predicate);

        auto n = geographies.size();
        auto* data = static_cast<PyObjectGeography*>(geographies.request().ptr);
        std::vector<py::ssize_t> input_idx;
        std::vector<py::ssize_t> tree_idx;
        for (py::ssize_t i = 0; i < n; i++) {
            auto results = query_one(*data[i].as_geog_ptr(), pred ? &*pred : nullptr);
            for (int t : results) {
                input_idx.push_back(i);
                tree_idx.push_back(static_cast<py::ssize_t>(t));
            }
        }

        return to_pairs_array(input_idx, tree_idx);
    }

    // Scalar nearest query: single Geography -> 1-d array of sorted tree
    // indices of the nearest geographies (optionally with their distance).
    py::object query_nearest(const Geography& geography,
                             std::optional<double> max_distance,
                             bool return_distance,
                             bool exclusive,
                             bool all_matches,
                             double radius) const {
        S1ChordAngle distance;
        auto results = query_nearest_one(
            geography, to_chord_angle(max_distance, radius), exclusive, all_matches, &distance);
        auto indices = to_index_array(results);
        if (!return_distance) {
            return indices;
        }

        auto n = results.size();
        py::array_t<double> distances(static_cast<py::ssize_t>(n));
        auto dist_data = distances.mutable_unchecked<1>();
        auto dist = distance.ToAngle().radians() * radius;
        for (std::size_t i = 0; i < n; i++) {
            dist_data(static_cast<py::ssize_t>(i)) = dist;
        }
        return py::make_tuple(indices, distances);
    }

    // Array nearest query: 1-d array of Geography -> (2, K) array of
    // (input index, tree index) pairs (optionally with their distances).
    py::object query_nearest(const py::array_t<PyObjectGeography>& geographies,
                             std::optional<double> max_distance,
                             bool return_distance,
                             bool exclusive,
                             bool all_matches,
                             double radius) const {
        if (geographies.ndim() != 1) {
            throw py::type_error(
                "query geography must be a Geography or a 1-dimensional array of Geography");
        }

        // checked up front (as the predicate is in the array ``query``) so
        // that an empty input array raises too rather than silently returning
        if (exclusive) {
            check_has_geographies("query_nearest with exclusive=True");
        }
        auto max_dist = to_chord_angle(max_distance, radius);
        auto n = geographies.size();
        auto* data = static_cast<PyObjectGeography*>(geographies.request().ptr);
        std::vector<py::ssize_t> input_idx;
        std::vector<py::ssize_t> tree_idx;
        std::vector<double> dists;
        for (py::ssize_t i = 0; i < n; i++) {
            S1ChordAngle distance;
            auto results = query_nearest_one(
                *data[i].as_geog_ptr(), max_dist, exclusive, all_matches, &distance);
            for (int t : results) {
                input_idx.push_back(i);
                tree_idx.push_back(static_cast<py::ssize_t>(t));
                dists.push_back(distance.ToAngle().radians() * radius);
            }
        }

        auto pairs = to_pairs_array(input_idx, tree_idx);
        if (!return_distance) {
            return pairs;
        }

        auto k = dists.size();
        py::array_t<double> distances(static_cast<py::ssize_t>(k));
        auto dist_data = distances.mutable_unchecked<1>();
        for (std::size_t j = 0; j < k; j++) {
            dist_data(static_cast<py::ssize_t>(j)) = dists[j];
        }
        return py::make_tuple(pairs, distances);
    }

private:
    // encoding header: magic (the bytes "SPIX", which put32 writes least
    // significant byte first), format version, flags
    static constexpr std::uint32_t kEncodingMagic = 0x58495053;
    // format version of the header below; only this exact value is accepted
    // when decoding, so any future format change is a bump here plus a
    // decoder branch
    static constexpr std::uint8_t kEncodingVersion = 1;
    // flags: the blob carries a per-geography block section
    static constexpr std::uint8_t kFlagHasGeographies = 1;
    static constexpr int kHeaderBytes =
        sizeof(kEncodingMagic) + sizeof(kEncodingVersion) + sizeof(std::uint8_t);
    // one fixed-width (offset, length) pair per geography, so that block j is
    // addressable without scanning the section
    static constexpr std::size_t kBlockTableEntryBytes = 2 * sizeof(std::uint64_t);
    // a block holds at least the geography type and empty bytes plus the
    // 4-byte tag written by s2geography::Geography::EncodeTagged
    static constexpr std::uint64_t kBlockMinBytes = 2 + 4;

    // Both indexes below borrow from the members declared above them and are
    // therefore declared last, so that they are destroyed first: members are
    // destroyed in reverse declaration order, and an S2ShapeIndex touches its
    // shapes while being torn down.

    // shape id -> tree index, filled by the constructor (built mode) or read
    // back out of the blob by ``from_encoded`` (encoded mode)
    std::vector<int> m_values;

    // built mode (constructed from an array of geographies)
    // m_geographies keeps the Geography objects alive (the index borrows
    // their S2Shapes)
    py::array_t<PyObjectGeography> m_geographies;
    // wrapped for its S2ShapeIndex only: the constructor fills m_values
    // itself, so this GeographyIndex's own shape id -> value table stays
    // empty and neither its value() nor its Iterator may be used
    std::unique_ptr<s2geog::GeographyIndex> m_index;

    // encoded mode (loaded with from_encoded); the encoded bytes are kept
    // alive here as EncodedS2ShapeIndex works directly on them
    std::string m_encoded;
    py::ssize_t m_num_geographies = 0;
    // whether the blob carries per-geography blocks, and where each block
    // lives inside ``m_encoded`` (absolute offset, length)
    bool m_has_geographies = false;
    std::vector<std::pair<std::size_t, std::size_t>> m_geog_blocks;
    // number of shapes per tree entry according to the id map (used to
    // cross-check reconstructed geographies against the encoded index), and
    // the first shape id of each entry's run (-1 for entries with no shape)
    std::vector<int> m_shape_counts;
    std::vector<int> m_first_shape;
    // lazily reconstructed Geography objects, one slot per tree entry: a slot
    // is filled on first touch and cached for the lifetime of the index
    // (mutable: caching only, doesn't affect the observable query results)
    mutable std::vector<py::object> m_decoded_slots;
    mutable std::size_t m_decoded_count = 0;
    // cached ``geometries`` array of the reconstructed objects
    mutable py::object m_decoded_geometries;
    // holds S2Shapes materialized from m_decoded_slots (fat) or decoded out
    // of m_encoded (thin): declared after both
    std::unique_ptr<EncodedS2ShapeIndex> m_encoded_index;

    // only used internally by ``from_encoded`` (encoded mode)
    SpatialIndex() = default;

    // Append the per-geography section to ``encoder``: a fixed-width
    // (offset, length) table (offsets relative to the first block, blocks
    // contiguous in tree order) followed by the blocks themselves. Each block
    // mirrors the Geography pickle tuple: the geography type (int8), the
    // empty flag (uint8) and the geography serialized with s2geography's
    // ``EncodeTagged``, so decoding goes through exactly the same code path
    // as unpickling.
    static void encode_geography_blocks(Encoder& encoder,
                                        const std::vector<const Geography*>& geogs) {
        std::vector<Encoder> blocks;
        blocks.reserve(geogs.size());
        std::size_t blocks_bytes = 0;
        for (const Geography* geog : geogs) {
            Encoder block;
            block.Ensure(2);
            using IntType = std::underlying_type_t<GeographyType>;
            block.put8(static_cast<unsigned char>(static_cast<IntType>(geog->geog_type())));
            block.put8(geog->is_empty() ? 1 : 0);
            // default options, as in Geography::encode (the pickle support):
            // exact vertices (FAST hint), no covering, no lazy decoding
            s2geog::EncodeOptions encode_opts;
            geog->geog().EncodeTagged(&block, encode_opts);
            blocks_bytes += block.length();
            blocks.push_back(std::move(block));
        }

        encoder.Ensure(kBlockTableEntryBytes * blocks.size() + blocks_bytes);
        std::uint64_t offset = 0;
        for (const auto& block : blocks) {
            encoder.put64(offset);
            encoder.put64(static_cast<std::uint64_t>(block.length()));
            offset += static_cast<std::uint64_t>(block.length());
        }
        for (const auto& block : blocks) {
            encoder.putn(block.base(), block.length());
        }
    }

    // Parse and validate the per-geography section (see
    // ``encode_geography_blocks``) at the decoder's current position, filling
    // ``m_geog_blocks``, and skip the decoder past it. Returns false on any
    // inconsistency.
    bool init_geography_blocks(Decoder& decoder, std::uint64_t num_geographies) {
        // ``from_encoded`` bounds num_geographies by what is left of the
        // buffer, so this product cannot overflow
        auto table_bytes = static_cast<std::size_t>(kBlockTableEntryBytes * num_geographies);
        if (decoder.avail() < table_bytes) {
            return false;
        }
        auto blocks_begin = decoder.pos() + table_bytes;
        // what is left for the blocks themselves once the table is consumed
        auto blocks_avail = static_cast<std::uint64_t>(decoder.avail() - table_bytes);
        m_geog_blocks.reserve(num_geographies);
        std::uint64_t expected_offset = 0;
        for (std::uint64_t j = 0; j < num_geographies; j++) {
            std::uint64_t offset = decoder.get64();
            std::uint64_t length = decoder.get64();
            // blocks are contiguous and in order: reject gaps, overlaps and
            // runt blocks
            if (offset != expected_offset || length < kBlockMinBytes || length > blocks_avail) {
                return false;
            }
            m_geog_blocks.emplace_back(blocks_begin + static_cast<std::size_t>(offset),
                                       static_cast<std::size_t>(length));
            expected_offset = offset + length;
            // bounded on every entry rather than once after the loop: the
            // running sum is free to wrap past 2^64 before a single check at
            // the end ever looks at it, and each block would then have been
            // recorded at an in-range offset with an out-of-range length
            if (expected_offset > blocks_avail) {
                return false;
            }
        }
        decoder.skip(static_cast<std::ptrdiff_t>(expected_offset));
        return true;
    }

    // Check that every shape id clipped by the decoded index cells is covered
    // by the value table (fat blobs only).
    //
    // A thin blob carries its own shape stream, so ``num_shape_ids()`` is an
    // independent count and the table size check in ``from_encoded`` is a real
    // cross-check. A fat blob's index is sized from
    // ``GeographyBlockShapeFactory::size()``, i.e. from the value table
    // itself, so that check compares the table with itself and nothing
    // reconciles the shape ids stored in the index cells with it. A cell
    // clipping a shape id past the end of the table makes
    // ``EncodedS2ShapeIndex::shape()`` index ``shapes_`` out of bounds
    // (unchecked), which segfaults inside the distance queries.
    //
    // This is one linear pass over the index cells at load, of the same order
    // as the value table and block table passes ``from_encoded`` already
    // makes. It decodes cells but never shapes, so no geography is
    // reconstructed here.
    bool check_index_shape_ids() const {
        auto num_shape_ids = static_cast<int>(m_values.size());
        int max_shape_id = -1;
        for (S2ShapeIndex::Iterator iter(&shape_index(), S2ShapeIndex::BEGIN); !iter.done();
             iter.Next()) {
            const S2ShapeIndexCell& cell = iter.cell();
            for (int k = 0; k < cell.num_clipped(); k++) {
                int shape_id = cell.clipped(k).shape_id();
                if (shape_id < 0 || shape_id >= num_shape_ids) {
                    return false;
                }
                max_shape_id = std::max(max_shape_id, shape_id);
            }
        }
        // exact, not just in range: ``encode`` writes one value per shape of
        // the index it serializes and every such shape is clipped into at
        // least one cell, so a table with entries past the last shape the
        // cells reference did not come from it (and would make
        // ``num_shape_ids()`` -- which is this table's size for a fat blob --
        // over-report the shapes the index actually has)
        return max_shape_id + 1 == num_shape_ids;
    }

    bool is_encoded() const {
        return m_encoded_index != nullptr;
    }

    // The operations that need the Geography objects work on a built index
    // and on one decoded from a blob written with the geographies included;
    // they raise on an index decoded from an index-only blob.
    void check_has_geographies(const std::string& what) const {
        if (is_encoded() && !m_has_geographies) {
            throw py::value_error(what +
                                  " is not supported for an index loaded from bytes encoded "
                                  "with include_geographies=False (encode with "
                                  "include_geographies=True to keep the Geography objects)");
        }
    }

    // Whether a block's geography type byte can describe what its tagged
    // encoding actually holds. ``Geography::decode`` takes the type byte at
    // face value -- as it does when unpickling, where it comes from a trusted
    // ``encode`` -- so nothing else reconciles the two: a relabelled block
    // would decode into a geography whose ``get_type_id`` disagrees with the
    // geography itself (a Polygon byte on a point block, say).
    //
    // A singular type and its MULTI* form share one kind, so this narrows the
    // byte to a pair rather than to a single value: swapping the two labels
    // stays undetectable here (it would take counting the parts, which is
    // ``extract_geog_properties``' job and is not reachable from a decoded
    // Geography). That is a wrong ``get_type_id`` label on a corrupt blob,
    // with no effect on the geometry or on any query -- the same latitude
    // unpickling a hand-made state tuple already has.
    static bool type_matches_kind(GeographyType geog_type, s2geog::GeographyKind kind) {
        switch (kind) {
            case s2geog::GeographyKind::POINT:
            case s2geog::GeographyKind::CELL_CENTER:
                return geog_type == GeographyType::Point || geog_type == GeographyType::MultiPoint;
            case s2geog::GeographyKind::POLYLINE:
                return geog_type == GeographyType::LineString ||
                       geog_type == GeographyType::MultiLineString;
            case s2geog::GeographyKind::POLYGON:
                return geog_type == GeographyType::Polygon ||
                       geog_type == GeographyType::MultiPolygon;
            case s2geog::GeographyKind::GEOGRAPHY_COLLECTION:
                return geog_type == GeographyType::GeometryCollection;
            default:
                // the kinds rejected just above; GeographyType::None belongs
                // with those, so it never matches either
                return false;
        }
    }

    // Reconstruct the Geography of tree entry ``j`` from its block, or return
    // the cached reconstruction. Decoding goes through Geography::decode (the
    // pickle path), so the objects are indistinguishable from unpickled ones.
    //
    // The caller must hold the GIL. The slot cache is a plain vector and the
    // counter a plain size_t, so the cache-hit path below is only safe
    // because every entry point reaches it with the GIL held (no query path
    // releases it) -- the ``gil_scoped_acquire`` guard sits on the decode
    // path and does not, and cannot, cover the slot read above it. It is
    // there so that the Python API calls of a decode still work should a
    // caller ever release the GIL around the C++ query work; making that
    // caller safe would take more than moving the guard up.
    Geography* decoded_geog(py::ssize_t i) const {
        auto j = static_cast<std::size_t>(i);
        py::object& slot = m_decoded_slots[j];
        if (!slot) {
            py::gil_scoped_acquire acquire;
            const auto& [offset, length] = m_geog_blocks[j];
            const char* block = m_encoded.data() + offset;
            auto geog_type = static_cast<std::int8_t>(static_cast<unsigned char>(block[0]));
            auto empty = static_cast<unsigned char>(block[1]);
            if (geog_type < -1 || geog_type > 6 || empty > 1) {
                throw py::value_error("invalid encoded SpatialIndex geography block");
            }
            auto encoded_geog = py::bytes(block + 2, length - 2);
            std::unique_ptr<Geography> geog;
            try {
                auto tuple = py::make_tuple(geog_type, empty != 0, encoded_geog);
                geog = std::make_unique<Geography>(Geography::decode(tuple));
            } catch (const py::error_already_set&) {
                throw;
            } catch (const std::exception&) {
                throw py::value_error("invalid encoded SpatialIndex geography block");
            }
            // ``encode`` only ever writes the kinds a spherely Geography can
            // wrap: the remaining ones are not just unexpected here, they are
            // unsound. UNINITIALIZED and SHAPE_INDEX have no WKT
            // representation (accessors raise "Unsupported Geography
            // subclass"), and an ENCODED_SHAPE_INDEX decodes lazily over the
            // buffer it was handed -- the temporary string inside
            // Geography::decode -- so it would outlive its own data.
            switch (geog->geog().kind()) {
                case s2geog::GeographyKind::UNINITIALIZED:
                case s2geog::GeographyKind::SHAPE_INDEX:
                case s2geog::GeographyKind::ENCODED_SHAPE_INDEX:
                    throw py::value_error("invalid encoded SpatialIndex geography block");
                default:
                    break;
            }
            if (!type_matches_kind(geog->geog_type(), geog->geog().kind())) {
                throw py::value_error("invalid encoded SpatialIndex geography block");
            }
            // the id map and the block must agree on the shape count, or the
            // reconstructed geography is not the one the index was built on
            if (geog->num_shapes() != m_shape_counts[j]) {
                throw py::value_error("invalid encoded SpatialIndex geography block");
            }
            auto decoded = PyObjectGeography::from_geog(std::move(geog));
            // everything between the ``!slot`` test and here runs
            // Python-visible work (py::bytes, Geography::decode, from_geog),
            // any of which can trigger a collection whose __del__ re-enters
            // this index and fills the same slot. The index may already have
            // cached an S2Shape borrowing that inner object, so overwriting
            // the slot would leave the cached shape pointing at a dropped
            // geography: keep the entry that got there first.
            if (!slot) {
                slot = std::move(decoded);
                m_decoded_count++;
            }
        }
        return static_cast<PyObjectGeography&>(slot).as_geog_ptr();
    }

    // Geography of tree entry ``i``: from the built array, or lazily
    // reconstructed from its encoded block. Predicate refinement and equality
    // run on these objects through the same code paths either way.
    Geography* candidate_geog(py::ssize_t i) const {
        if (is_encoded()) {
            return decoded_geog(i);
        }
        auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
        return data[i].as_geog_ptr();
    }

    // Produce shape ``shape_id`` of a fat blob's index by reconstructing (or
    // reusing) the geography that owns it. The returned shape borrows the
    // geography's data, which stays alive in the slot cache for the lifetime
    // of this index. Called from GeographyBlockShapeFactory inside query
    // execution, always with the GIL held (see decoded_geog).
    std::unique_ptr<S2Shape> materialize_shape(int shape_id) const {
        auto j = m_values.at(static_cast<std::size_t>(shape_id));
        Geography* geog = decoded_geog(j);
        // decoded_geog checked the block against the id map's shape count,
        // so the run arithmetic below cannot leave the geography's range
        int sub = shape_id - m_first_shape[static_cast<std::size_t>(j)];
        return geog->geog().Shape(sub);
    }

    // ShapeFactory backing the EncodedS2ShapeIndex of a fat blob: there is
    // no second copy of the shape data in the blob, shapes are pulled from
    // the per-geography blocks instead. The owner pointer is stable
    // (SpatialIndex is immovable) and outlives the encoded index holding the
    // cloned factory, which is declared last among the members precisely so
    // that it is torn down before everything it borrows.
    class GeographyBlockShapeFactory : public S2ShapeIndex::ShapeFactory {
    public:
        explicit GeographyBlockShapeFactory(const SpatialIndex* owner) : m_owner(owner) {}

        int size() const override {
            return static_cast<int>(m_owner->m_values.size());
        }

        std::unique_ptr<S2Shape> operator[](int shape_id) const override {
            return m_owner->materialize_shape(shape_id);
        }

        std::unique_ptr<ShapeFactory> Clone() const override {
            return std::make_unique<GeographyBlockShapeFactory>(*this);
        }

    private:
        const SpatialIndex* m_owner;
    };

    const S2ShapeIndex& shape_index() const {
        if (is_encoded()) {
            return *m_encoded_index;
        }
        return m_index->ShapeIndex();
    }

    // tree index of the geography that owns the given shape
    int tree_value(int shape_id) const {
        // the constructor covers every shape of a built index and
        // ``from_encoded`` checks that the decoded table covers every shape
        // id; ``at`` keeps a mismatch a Python exception rather than a bad
        // read
        return m_values.at(static_cast<std::size_t>(shape_id));
    }

    std::optional<PredicateFunc> make_predicate(const std::optional<std::string>& name) const {
        if (!name.has_value()) {
            return std::nullopt;
        }
        check_has_geographies("query with a predicate");
        return get_predicate(*name);
    }

    // Collect the tree indices of the shapes in the index cells overlapping
    // ``cell_id`` (mirrors s2geography::GeographyIndex::Iterator::Query,
    // which is bound to the mutable index of the built mode).
    void query_cell(S2ShapeIndex::Iterator& iter,
                    const S2CellId& cell_id,
                    std::unordered_set<int>& hits) const {
        auto add_cell_shapes = [&]() {
            const S2ShapeIndexCell& index_cell = iter.cell();
            for (int k = 0; k < index_cell.num_clipped(); k++) {
                hits.insert(tree_value(index_cell.clipped(k).shape_id()));
            }
        };

        S2CellRelation relation = iter.Locate(cell_id);
        if (relation == S2CellRelation::INDEXED) {
            // the index has this cell (or an ancestor of it)
            add_cell_shapes();
        } else if (relation == S2CellRelation::SUBDIVIDED) {
            // the index has child cells of ``cell_id`` (the iterator is
            // positioned at the first one): visit them all
            while (!iter.done() && cell_id.contains(iter.id())) {
                add_cell_shapes();
                iter.Next();
            }
        }
        // else: DISJOINT (do nothing)
    }

    // Return the sorted tree indices whose cells overlap the query geography,
    // optionally refined by ``pred`` (predicate(query, candidate)).
    std::vector<int> query_one(const Geography& query_geog, const PredicateFunc* pred) const {
        std::unordered_set<int> hits;

        auto region = query_geog.geog().Region();
        S2RegionCoverer coverer;
        std::vector<S2CellId> covering;
        coverer.GetCovering(*region, &covering);

        S2ShapeIndex::Iterator iter(&shape_index());
        for (const S2CellId& cell_id : covering) {
            query_cell(iter, cell_id, hits);
        }

        std::vector<int> results;
        if (pred == nullptr) {
            results.assign(hits.begin(), hits.end());
        } else {
            const auto& query_index = query_geog.geog_index();
            for (int candidate : hits) {
                auto* cand_geog = candidate_geog(candidate);
                if ((*pred)(query_index, cand_geog->geog_index())) {
                    results.push_back(candidate);
                }
            }
        }

        std::sort(results.begin(), results.end());
        return results;
    }

    // Convert a distance in meters (given the radius) to a chord angle.
    static std::optional<S1ChordAngle> to_chord_angle(std::optional<double> distance,
                                                      double radius) {
        // the radius scales both the returned distances and ``max_distance``,
        // so it has to be checked even when there is no distance bound
        if (!std::isfinite(radius) || radius <= 0) {
            throw py::value_error("radius must be a finite value greater than 0");
        }
        if (!distance.has_value()) {
            return std::nullopt;
        }
        // NaN would compare false here and silently mean "unbounded"
        if (!std::isfinite(*distance) || *distance <= 0) {
            throw py::value_error("max_distance must be a finite value greater than 0");
        }
        auto bound = S1ChordAngle(S1Angle::Radians(*distance / radius));
        // the meters -> radians -> chord angle conversion here and the chord
        // angle -> radians -> meters conversion of the returned distances do
        // not round trip exactly (they disagree by a couple of ULPs), so a
        // distance reported by ``query_nearest`` would otherwise be rejected
        // as a max_distance about a quarter of the time. Widen the bound by a
        // few ULPs to keep it inclusive (S1ChordAngle::set_inclusive_max_distance
        // only bumps it by one). FromLength2 clamps to a straight angle.
        return S1ChordAngle::FromLength2(bound.length2() *
                                         (1 + 8 * std::numeric_limits<double>::epsilon()));
    }

    // Return the sorted tree indices of every geography whose distance to the
    // target is exactly ``dist`` (``dist`` must be the minimum distance over
    // the geographies not skipped, so that no geography has an edge in
    // ``(0, dist)``).
    std::vector<int> collect_at_distance(S2ClosestEdgeQuery& query,
                                         S2ClosestEdgeQuery::ShapeIndexTarget& target,
                                         S1ChordAngle dist) const {
        query.mutable_options()->set_max_results(S2ClosestEdgeQuery::Options::kMaxMaxResults);
        query.mutable_options()->set_inclusive_max_distance(dist);
        std::unordered_set<int> hits;
        for (const auto& result : query.FindClosestEdges(&target)) {
            if (result.distance() == dist) {
                hits.insert(tree_value(result.shape_id()));
            }
        }
        std::vector<int> results(hits.begin(), hits.end());
        std::sort(results.begin(), results.end());
        return results;
    }

    // ``collect_at_distance``, reduced to the lowest tree index if
    // ``all_matches`` is false: which tie is returned is unspecified by the
    // API but is picked deterministically (the results are sorted).
    std::vector<int> nearest_at_distance(S2ClosestEdgeQuery& query,
                                         S2ClosestEdgeQuery::ShapeIndexTarget& target,
                                         S1ChordAngle dist,
                                         bool all_matches) const {
        auto results = collect_at_distance(query, target, dist);
        if (!all_matches && results.size() > 1) {
            results.resize(1);
        }
        return results;
    }

    // Return the sorted tree indices of the geographies nearest to
    // ``query_geog`` (every tie at the minimum distance if ``all_matches``,
    // otherwise the lowest of them), storing the minimum distance in
    // ``*distance`` (Infinity if there is no result). Geographies farther
    // than ``max_distance`` (inclusive) are never returned; if ``exclusive``,
    // neither are geographies equal to ``query_geog``.
    std::vector<int> query_nearest_one(const Geography& query_geog,
                                       std::optional<S1ChordAngle> max_distance,
                                       bool exclusive,
                                       bool all_matches,
                                       S1ChordAngle* distance) const {
        *distance = S1ChordAngle::Infinity();
        if (exclusive) {
            check_has_geographies("query_nearest with exclusive=True");
        }
        if (query_geog.is_empty()) {
            return {};
        }

        S2ClosestEdgeQuery query(&shape_index());
        // count the interior of indexed polygons (resp. of the query
        // geography) as distance zero, consistent with spherely.distance
        query.mutable_options()->set_include_interiors(true);
        if (max_distance.has_value()) {
            query.mutable_options()->set_inclusive_max_distance(*max_distance);
        }
        S2ClosestEdgeQuery::ShapeIndexTarget target(&query_geog.geog_index().ShapeIndex());
        target.set_include_interiors(true);

        auto closest = query.FindClosestEdge(&target);
        if (closest.is_empty()) {
            return {};
        }
        auto min_dist = closest.distance();

        if (!exclusive || min_dist > S1ChordAngle::Zero()) {
            // geographies equal to the query would be at distance zero, so
            // nothing needs to be excluded here
            *distance = min_dist;
            return nearest_at_distance(query, target, min_dist, all_matches);
        }

        // exclusive with matches at distance zero: split those into
        // geographies equal to the query (excluded) and others (returned)
        auto zero_hits = collect_at_distance(query, target, S1ChordAngle::Zero());
        auto equals_pred = get_predicate("equals");
        const auto& query_index = query_geog.geog_index();
        std::vector<int> non_equal;
        for (int t : zero_hits) {
            if (!equals_pred(query_index, candidate_geog(t)->geog_index())) {
                non_equal.push_back(t);
            }
        }
        if (!non_equal.empty()) {
            *distance = S1ChordAngle::Zero();
            if (!all_matches) {
                return {non_equal.front()};  // sorted: the lowest tree index
            }
            return non_equal;
        }

        // every geography at distance zero equals the query: look for the
        // nearest one at a distance > 0. Every edge of an equal geography is
        // at distance zero (equal geographies cover each other), so the first
        // result at a positive distance -- results are sorted by distance --
        // gives the minimum distance over the non-equal geographies.
        if (max_distance.has_value()) {
            query.mutable_options()->set_inclusive_max_distance(*max_distance);
        } else {
            query.mutable_options()->set_max_distance(S1ChordAngle::Infinity());
        }
        constexpr int kMaxResults = S2ClosestEdgeQuery::Options::kMaxMaxResults;
        for (int max_results = 16;;) {
            query.mutable_options()->set_max_results(max_results);
            auto results = query.FindClosestEdges(&target);
            for (const auto& result : results) {
                if (result.distance() > S1ChordAngle::Zero()) {
                    *distance = result.distance();
                    return nearest_at_distance(query, target, result.distance(), all_matches);
                }
            }
            if (results.size() < static_cast<std::size_t>(max_results) ||
                max_results == kMaxResults) {
                // all edges within max_distance seen, none at distance > 0
                return {};
            }
            // saturate instead of overflowing (which would be undefined)
            max_results = max_results > kMaxResults / 2 ? kMaxResults : max_results * 2;
        }
    }

    static py::array_t<py::ssize_t> to_index_array(const std::vector<int>& indices) {
        auto n = indices.size();
        py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(n));
        auto out_data = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < n; i++) {
            out_data(static_cast<py::ssize_t>(i)) = static_cast<py::ssize_t>(indices[i]);
        }
        return out;
    }

    static py::array_t<py::ssize_t> to_pairs_array(const std::vector<py::ssize_t>& input_idx,
                                                   const std::vector<py::ssize_t>& tree_idx) {
        auto k = input_idx.size();
        py::array_t<py::ssize_t> out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(k)});
        auto out_data = out.mutable_unchecked<2>();
        for (std::size_t j = 0; j < k; j++) {
            out_data(0, static_cast<py::ssize_t>(j)) = input_idx[j];
            out_data(1, static_cast<py::ssize_t>(j)) = tree_idx[j];
        }
        return out;
    }
};

void init_spatial_index(py::module& m) {
    py::class_<SpatialIndex>(m, "SpatialIndex", R"pbdoc(
        A spatial index for a collection of geographies.

        The index allows fast retrieval of the geographies that may interact
        with a given geography, e.g., for accelerating spatial joins. It is
        built on top of s2geometry's shape index and is conceptually similar to
        shapely's ``STRtree``.

        The underlying shape index is built lazily: the constructor only queues
        the geographies and the first query pays for the build. Use
        :py:meth:`SpatialIndex.build` to do that work up front instead.

        Parameters
        ----------
        geographies : array_like
            An array (or other sequence) of :py:class:`Geography` objects. Empty
            geographies are indexed but never returned by queries.

    )pbdoc")
        .def(py::init<const py::array_t<PyObjectGeography>&>(),
             py::arg("geographies"),
             "__init__(self, geographies)")
        .def("build",
             &SpatialIndex::build,
             py::arg("tmp_memory_budget") = py::none(),
             R"pbdoc(build(tmp_memory_budget=None)

        Force the deferred index build, optionally under a raised temporary
        memory budget.

        The index is built lazily, so without this call the first query pays
        the whole build cost. Calling it is optional, never changes the result
        of a query made with a ``predicate``, and does nothing once the index
        is built (see :py:attr:`SpatialIndex.is_built`) -- including on an
        index loaded with :py:meth:`SpatialIndex.from_encoded`, which has no
        build to force.

        Parameters
        ----------
        tmp_memory_budget : int, optional
            Temporary memory budget for this build, in bytes. s2geometry builds
            the index in batches sized to fit it, and many batches cost more
            than one, so the knob is only useful *raised*; about 226 bytes per
            edge is enough to build in a single batch. Must be strictly
            positive, and must fit in a signed 64-bit integer (larger values
            raise ``TypeError``). The previous value is put back when the call
            returns, so the override does not leak into any later build.

            ``None`` does *not* mean "no budget": it leaves s2geometry's
            setting untouched (104857600 bytes, i.e. 100 MB, unless something
            else in the process changed it), so the build still batches.

        Notes
        -----
        Peak resident memory can rise by roughly the budget granted, and
        nothing is granted unless a budget is passed.

        The budget changes where s2geometry places its batch boundaries, and so
        how it subdivides the index. A query made without a ``predicate`` can
        therefore return a different candidate set under a different budget --
        candidates may be added *or* dropped, and the sets for two budgets need
        not be nested. What does not change is that the candidate set is always
        a superset of the true matches, so any ``predicate`` refines it to the
        same answer.

        This call holds the GIL for its whole duration -- minutes, for a large
        collection -- and the budget is a process-wide s2geometry setting for
        that duration.

        Examples
        --------
        >>> import spherely
        >>> index = spherely.SpatialIndex(geographies)
        >>> index.build(tmp_memory_budget=4 * 1024**3)  # 4 GiB

    )pbdoc")
        .def("__len__", &SpatialIndex::size)
        .def_property_readonly("is_built",
                               &SpatialIndex::is_built,
                               R"pbdoc(Whether the underlying shape index has been built.

        ``False`` between the constructor and the first
        :py:meth:`SpatialIndex.build` or query, ``True`` afterwards. An index
        over an empty collection is ``True`` from the start, and so is one
        loaded with :py:meth:`SpatialIndex.from_encoded` (which decodes on
        demand and has nothing to build).

    )pbdoc")
        .def_property_readonly("geometries",
                               &SpatialIndex::geometries,
                               "The array of geographies in the index (in input order).")
        .def_property_readonly("_decoded_count",
                               &SpatialIndex::decoded_count,
                               "Number of geographies reconstructed so far by an index loaded "
                               "from encoded bytes (testing/introspection only).")
        .def("encode",
             &SpatialIndex::encode,
             py::arg("include_geographies") = true,
             R"pbdoc(encode(include_geographies=True)

        Serialize the index to bytes.

        By default the Geography objects themselves are serialized along with
        the index structure, so that :py:meth:`SpatialIndex.from_encoded`
        restores a fully functional index. The shape data is stored only once
        (inside the serialized geographies, which the decoded index reads its
        shapes from). With ``include_geographies=False`` an index-only blob
        is written instead. It is the smaller of the two, by an amount that
        depends on the geometry mix: in measurements the default blob ran
        from about 1.1x the index-only one (polygon-heavy indexes) to about
        2x (small point-only ones). The loaded index does not support the
        operations that need the Geography objects, and its distance queries
        decode individual shapes rather than whole geographies (see
        :py:meth:`SpatialIndex.from_encoded`).

        .. warning::
           The encoded bytes are not a portable interchange format: past a
           small spherely header they are s2geometry's (and s2geography's)
           own encoding, which is tied to the library versions spherely was
           built against. The per-geography blocks in particular round-trip
           through ``s2geography::Geography::EncodeTagged`` /
           ``DecodeTagged``, which s2geography labels EXPERIMENTAL. A blob
           written by one build is not guaranteed to be readable by a build
           linked against a different s2geometry (loading it raises
           ``ValueError`` rather than returning wrong results). Encode and
           decode with the same spherely build, and treat the bytes as a
           cache, not as an archival format.

        Parameters
        ----------
        include_geographies : bool, default True
            If True, serialize the Geography objects along with the index so
            that the decoded index supports every operation. If False, write
            the smaller index-only blob.

        Returns
        -------
        bytes
            The encoded index.

    )pbdoc")
        .def_static("from_encoded",
                    &SpatialIndex::from_encoded,
                    py::arg("encoded"),
                    R"pbdoc(from_encoded(encoded)

        Load an index serialized with :py:meth:`SpatialIndex.encode`.

        The returned index is a lazy view on the encoded bytes: loading it is
        near-instant regardless of the index size, and the index cells are
        only decoded on demand by queries. For bytes written with
        ``include_geographies=True`` (the default) the Geography of a tree
        entry is reconstructed the first time an operation touches that
        entry: by predicate refinement of a candidate, by a distance query
        (``query_nearest``) examining one of the entry's shapes, or by the
        ``geometries`` property (which reconstructs all entries). Candidate
        queries (``query`` without a predicate) only walk the index cells
        and reconstruct nothing. Reconstruction of an entry decodes that
        geography completely (through the same code path as unpickling a
        Geography) and the result is cached, so each entry pays this cost at
        most once and untouched entries never pay it.

        ``query_nearest`` is the costly one: it reconstructs every entry
        whose shapes the distance search examines, and it reconstructs them
        whole even though it only needs one shape of each. That is a few
        tens of geographies per call -- in measurements on point indexes of
        100 to 20,000 entries, between about 10 and 80, so near-constant in
        the size of the index rather than proportional to it -- against the
        single shapes an index-only blob decodes for the same query.

        The decoded index supports every operation with the same results --
        including the same integer indices -- as the index it was encoded
        from. For bytes written with ``include_geographies=False`` the
        Geography objects are not available, so the ``geometries`` property,
        ``query`` with a predicate and ``query_nearest`` with
        ``exclusive=True`` raise ``ValueError``.

        Only bytes written by :py:meth:`SpatialIndex.encode` from the same
        spherely build are supported -- see the warning there about the
        format's coupling to s2geometry. Corrupt or foreign bytes raise
        ``ValueError``.

        Parameters
        ----------
        encoded : bytes-like
            An index serialized with :py:meth:`SpatialIndex.encode`, as any
            contiguous bytes-like object (``bytes``, ``bytearray``,
            ``memoryview``, ...). The bytes are copied.

        Returns
        -------
        :py:class:`SpatialIndex`
            The loaded index.

    )pbdoc")
        .def("query",
             py::overload_cast<const Geography&, std::optional<std::string>>(&SpatialIndex::query,
                                                                             py::const_),
             py::arg("geography"),
             py::arg("predicate") = py::none(),
             R"pbdoc(query(geography, predicate=None)

        Return the integer indices of all geographies in the index whose cells
        overlap the given geography (a coarse candidate set), optionally refined
        by a spatial predicate.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            The geography or geographies to query.
        predicate : str, optional
            If provided, only return candidates for which
            ``predicate(geography, indexed_geography)`` is True. One of
            "intersects", "within", "contains", "covers", "covered_by",
            "touches" or "equals".

        Returns
        -------
        ndarray
            If ``geography`` is a scalar, a 1-d array of (sorted) indices into
            the index. If ``geography`` is an array, a ``(2, N)`` array where the
            first row holds the input geography indices and the second row the
            matching index (tree) indices.

    )pbdoc")
        .def("query",
             py::overload_cast<const py::array_t<PyObjectGeography>&, std::optional<std::string>>(
                 &SpatialIndex::query, py::const_),
             py::arg("geography"),
             py::arg("predicate") = py::none())
        .def(
            "query_nearest",
            py::overload_cast<const Geography&, std::optional<double>, bool, bool, bool, double>(
                &SpatialIndex::query_nearest, py::const_),
            py::arg("geography"),
            py::arg("max_distance") = py::none(),
            py::arg("return_distance") = false,
            py::arg("exclusive") = false,
            py::arg("all_matches") = true,
            py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS,
            R"pbdoc(query_nearest(geography, max_distance=None, return_distance=False, exclusive=False, all_matches=True, radius=spherely.EARTH_RADIUS_METERS)

        Return the integer indices of the geographies in the index that are
        nearest to the given geography, similar to shapely's
        ``STRtree.query_nearest``.

        The distance to a geography is zero if the query geography intersects
        it (including polygon interiors), consistent with
        :py:func:`spherely.distance`. Empty geographies (in the index or as
        query) are never matched.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            The geography or geographies to query.
        max_distance : float, optional
            The maximum distance (in meters, inclusive) within which to query
            for the nearest geographies. Must be a finite value greater than 0
            (pass None, the default, for no bound).
        return_distance : bool, default False
            If True, also return the distance(s) to the nearest geographies.
        exclusive : bool, default False
            If True, geographies in the index that are equal to the query
            geography are not returned.
        all_matches : bool, default True
            If True, return all geographies tied at the minimum distance. If
            False, return only one of them: the one with the lowest index
            (shapely leaves the choice arbitrary, spherely makes it
            deterministic).
        radius : float, optional
            Radius of Earth in meters, default 6,371,010. Used to scale the
            returned distances and to interpret ``max_distance``. Must be a
            finite value greater than 0.

        Returns
        -------
        ndarray or tuple of ndarray
            If ``geography`` is a scalar, a 1-d array of (sorted) indices into
            the index. If ``geography`` is an array, a ``(2, K)`` array where
            the first row holds the input geography indices and the second row
            the matching index (tree) indices. If ``return_distance`` is True,
            a ``(indices, distances)`` tuple where ``distances`` holds the
            distance in meters for each match.

    )pbdoc")
        .def("query_nearest",
             py::overload_cast<const py::array_t<PyObjectGeography>&,
                               std::optional<double>,
                               bool,
                               bool,
                               bool,
                               double>(&SpatialIndex::query_nearest, py::const_),
             py::arg("geography"),
             py::arg("max_distance") = py::none(),
             py::arg("return_distance") = false,
             py::arg("exclusive") = false,
             py::arg("all_matches") = true,
             py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS);

    // Not public API: exposed so that the test suite can move the process-wide
    // budget off its default and assert what SpatialIndex.build() puts back.
    m.def("_s2_tmp_memory_budget",
          &get_s2_tmp_memory_budget,
          "Return s2geometry's current index-build temporary memory budget, in bytes.");
    m.def("_set_s2_tmp_memory_budget",
          &set_s2_tmp_memory_budget,
          py::arg("bytes"),
          "Set s2geometry's index-build temporary memory budget, in bytes.");
}
