// Pinhole rays in Luanti XYZ. Grid storage remains ENU (x, z, y).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr int size = 48;
constexpr int cells = size * size * size;
constexpr double low = -24.5;
constexpr double high = 23.5;
constexpr double eps = 1e-8;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr uint16_t empty_id = 65535;
constexpr uint8_t empty_face = 255;
using Vec = std::array<double, 3>;
using Cell = std::array<int, 3>;

Vec operator+(const Vec &a, const Vec &b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vec operator-(const Vec &a, const Vec &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec operator*(const Vec &a, double b) {
    return {a[0] * b, a[1] * b, a[2] * b};
}
double dot(const Vec &a, const Vec &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vec cross(const Vec &a, const Vec &b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
Vec unit(const Vec &v) { return v * (1.0 / std::sqrt(dot(v, v))); }
int flat(const Cell &p) { return (p[0] * size + p[2]) * size + p[1]; }

struct Hit {
    double t = inf;
    int owner = -1;
    int face = -1;
};

bool slab(const Vec &origin, const Vec &ray, const Vec &min, const Vec &max,
          double &entry, double &exit, int &face) {
    entry = -inf;
    exit = inf;
    face = -1;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(ray[axis]) < eps) {
            if (origin[axis] < min[axis] || origin[axis] > max[axis])
                return false;
            continue;
        }
        double t0 = (min[axis] - origin[axis]) / ray[axis];
        double t1 = (max[axis] - origin[axis]) / ray[axis];
        const bool positive = t0 < t1;
        if (!positive)
            std::swap(t0, t1);
        if (t0 > entry) {
            entry = t0;
            face = 2 * axis + positive;
        }
        exit = std::min(exit, t1);
        if (entry >= exit)
            return false;
    }
    return true;
}

struct Shape {
    int owner;
    int kind; // 0 box, 1 two-sided triangle, 2 outward triangle
    Vec a, b, c;
    Vec u, v, normal;

    explicit Shape(const float *row)
        : owner(int(row[0])), kind(int(row[1])),
          a{row[2], row[3], row[4]}, b{row[5], row[6], row[7]},
          c{row[8], row[9], row[10]}, u(b - a), v(c - a), normal(cross(u, v)) {}

    Hit entry(const Vec &origin, const Vec &ray) const {
        if (kind == 0) {
            double start, end;
            int face;
            return slab(origin, ray, a, b, start, end, face) ? Hit{start, owner, face} : Hit{};
        }
        const double front = dot(normal, ray);
        if (std::abs(front) < eps || (kind == 2 && front > 0))
            return {};
        const auto relative = origin - a;
        const auto p = cross(ray, v);
        const auto q = cross(relative, u);
        const double inverse = -1 / front;
        const double s = dot(relative, p) * inverse;
        const double t = dot(ray, q) * inverse;
        if (s < -eps || t < -eps || s + t > 1 + eps)
            return {};
        int axis = 0;
        for (int i = 1; i < 3; ++i)
            if (std::abs(normal[i]) > std::abs(normal[axis]))
                axis = i;
        return {dot(v, q) * inverse, owner, 2 * axis + (normal[axis] * front > 0)};
    }
};

struct Scene {
    const uint16_t *ids;
    const uint8_t *params, *cubes;
    std::vector<Shape> shapes;
    std::vector<std::vector<int>> entries{cells};

    Scene(const uint16_t *nodes, const uint8_t *param2, const uint8_t *full,
          const float *geometry, int n) : ids(nodes), params(param2), cubes(full) {
        shapes.reserve(n);
        for (int i = 0; i < n; ++i) {
            const auto &shape = shapes.emplace_back(geometry + 11 * i);
            Cell begin, end;
            for (int axis = 0; axis < 3; ++axis) {
                double a = std::min(shape.a[axis], shape.b[axis]);
                double b = std::max(shape.a[axis], shape.b[axis]);
                if (shape.kind) {
                    a = std::min(a, shape.c[axis]);
                    b = std::max(b, shape.c[axis]);
                }
                begin[axis] = std::max(0, int(std::floor(a - low - eps)));
                end[axis] = std::min(size - 1, int(std::floor(b - low + eps)));
            }
            for (int x = begin[0]; x <= end[0]; ++x)
                for (int z = begin[2]; z <= end[2]; ++z)
                    for (int y = begin[1]; y <= end[1]; ++y)
                        entries[flat({x, y, z})].push_back(i);
        }
    }

    std::array<Hit, 3> trace(const Vec &origin, const Vec &ray) const {
        std::array<Hit, 3> result{};
        double entry, exit;
        int face;
        if (!slab(origin, ray, {low, low, low}, {high, high, high}, entry, exit, face))
            return result;
        // The engine's near plane is 0.1 voxels along the camera forward axis.
        double t = std::max(entry, 0.1);
        if (t >= exit)
            return result;
        if (t > entry)
            face = -1;
        const auto pos = origin + ray * (t + eps);
        Cell p, step;
        Vec next, period;
        for (int axis = 0; axis < 3; ++axis) {
            p[axis] = std::clamp(int(std::floor(pos[axis] - low)), 0, size - 1);
            step[axis] = ray[axis] >= 0 ? 1 : -1;
            if (std::abs(ray[axis]) < eps) {
                next[axis] = period[axis] = inf;
            } else {
                next[axis] = (low + p[axis] + (step[axis] > 0) - origin[axis]) / ray[axis];
                period[axis] = std::abs(1 / ray[axis]);
            }
        }
        int count = 0;
        while (count < 3 && t < exit - eps) {
            const double end = std::min({next[0], next[1], next[2], exit});
            const int index = flat(p);
            std::array<Hit, 3> local{};
            const auto insert = [&](Hit hit) {
                if (!(hit.t >= t - eps && hit.t <= end + eps))
                    return;
                // A voxel can contribute one entry even if several primitives are hit.
                for (int i = 0; i < count; ++i)
                    if (result[i].owner == hit.owner)
                        return;
                for (auto &value : local) {
                    if (value.owner == hit.owner) {
                        if (value.t <= hit.t)
                            return;
                        value = hit;
                        std::sort(local.begin(), local.end(), [](Hit a, Hit b) { return a.t < b.t; });
                        return;
                    }
                }
                for (auto &value : local)
                    if (hit.t < value.t)
                        std::swap(hit, value);
            };
            if (face >= 0 && cubes[index])
                insert({t, index, face});
            for (const int candidate : entries[index])
                insert(shapes[candidate].entry(origin, ray));
            for (const auto &hit : local)
                if (hit.owner >= 0 && count < 3)
                    result[count++] = hit;

            if (end >= exit - eps)
                break;
            t = end;
            face = -1;
            for (int axis = 0; axis < 3; ++axis) {
                if (next[axis] > end + eps)
                    continue;
                p[axis] += step[axis];
                next[axis] += period[axis];
                if (face < 0)
                    face = 2 * axis + (step[axis] > 0);
            }
            if (std::any_of(p.begin(), p.end(), [](int x) { return x < 0 || x >= size; }))
                break;
        }
        return result;
    }
};
} // namespace

extern "C" {
void *vox2pix_scene(const uint16_t *nodes, const uint8_t *params, const uint8_t *cubes,
                    const float *geometry, int count) {
    try {
        return new Scene(nodes, params, cubes, geometry, count);
    } catch (...) {
        return nullptr;
    }
}

void vox2pix_destroy(void *scene) {
    delete static_cast<Scene *>(scene);
}

void vox2pix_render(const void *scene, const double *camera, const double *direction,
                    double fov_x, double fov_y, int w, int h, uint16_t *nodes,
                    uint8_t *params, uint8_t *faces) {
    const auto &world = *static_cast<const Scene *>(scene);
    const Vec origin{camera[0], camera[1], camera[2]};
    const Vec forward = unit({direction[0], direction[1], direction[2]});
    // Irrlicht's unrolled look-at basis, with +Z as forward and +X as screen right at yaw 0.
    const Vec right = unit(cross({0., 1., 0.}, forward)) * std::tan(fov_x / 2);
    const Vec up = unit(cross(forward, right)) * std::tan(fov_y / 2);
    const auto pixels = w * h;
    for (int y = 0; y < h; ++y) {
        const Vec row = forward + up * (1 - 2 * (y + 0.5) / h);
        for (int x = 0; x < w; ++x) {
            const Vec ray = row + right * (2 * (x + 0.5) / w - 1);
            const auto hits = world.trace(origin, ray);
            for (int layer = 0; layer < 3; ++layer) {
                const int out = layer * pixels + y * w + x;
                const auto &hit = hits[layer];
                nodes[out] = hit.owner < 0 ? empty_id : world.ids[hit.owner];
                params[out] = hit.owner < 0 ? 0 : world.params[hit.owner];
                faces[out] = hit.owner < 0 ? empty_face : hit.face;
            }
        }
    }
}
}
