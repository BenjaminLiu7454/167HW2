[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(
    uint tid : SV_DispatchThreadID,
    uniform uint num_pixels,
    RWStructuredBuffer<uint> depth_buffer,
    RWStructuredBuffer<uint> triangle_id_buffer)
{
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}


[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;

    float4 p = mul(MVP, float4(vertices[i], 1.0));

    projected_vertices[i] = p.xyz / p.w;
}



float cross2(float2 a, float2 b){
    return a.x * b.y - a.y * b.x;
}


float triangle_area(float2 p0, float2 p1, float2 p2){
    return cross2(p1 - p0, p2 - p0);
}



bool compute_barycentrics(float2 p, float2 p0, float2 p1, float2 p2, out float3 bary){
    float area = triangle_area(p0, p1, p2);

    if (abs(area) < 1e-8){
        bary = float3(0.0, 0.0, 0.0);
        return false;
    }

    bary.x = cross2(p1 - p, p2 - p) / area;
    bary.y = cross2(p2 - p, p0 - p) / area;
    bary.z = cross2(p0 - p, p1 - p) / area;

    return bary.x >= 0.0 &&
           bary.y >= 0.0 &&
           bary.z >= 0.0;
}


// Retrieves the projected vertices for a triangle
bool get_triangle(uint triangle_index, StructuredBuffer<uint3> faces, StructuredBuffer<float3> projected_vertices, out float3 v0, out float3 v1, out float3 v2){
    uint3 face = faces[triangle_index];

    v0 = projected_vertices[face.x];
    v1 = projected_vertices[face.y];
    v2 = projected_vertices[face.z];

    float area = triangle_area(v0.xy, v1.xy, v2.xy);

    return abs(area) >= 1e-8;
}


// Calculates a triangle's pixel bounding box and clips it to the image boundaries
bool get_pixel_bounds(float3 v0, float3 v1, float3 v2, uint width, uint height, out int2 lo, out int2 hi){
    float2 min_p = min(v0.xy, min(v1.xy, v2.xy));
    float2 max_p = max(v0.xy, max(v1.xy, v2.xy));

    // Convert the bounding box coordinates explicitly to integers.
    lo = int2(max(int(floor(min_p.x)), 0), max(int(floor(min_p.y)), 0));

    hi = int2(min(int(ceil(max_p.x)), int(width) - 1),min(int(ceil(max_p.y)), int(height) - 1));

    return lo.x <= hi.x && lo.y <= hi.y;
}


//Rasterize triangle depths
[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
    uint tid : SV_DispatchThreadID,
    uniform uint num_triangles,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    RWStructuredBuffer<uint> depth_buffer)
{
    if (tid >= num_triangles)
        return;

    float3 v0, v1, v2;

    if (!get_triangle(tid, faces, projected_vertices, v0, v1, v2)){
        return;
    }

    int2 lo, hi;

    if (!get_pixel_bounds(v0, v1, v2, width, height, lo, hi)){
        return;
    }

    for (int y = lo.y; y <= hi.y; ++y)
    {
        for (int x = lo.x; x <= hi.x; ++x)
        {
            // Sample the pixel center
            float2 p = float2(
                float(x) + 0.5,
                float(y) + 0.5
            );

            float3 bary;

            if (!compute_barycentrics(
                p,
                v0.xy,
                v1.xy,
                v2.xy,
                bary))
            {
                continue;
            }

            // Projected z stores inverse depth.
            // Interpolate inverse depth linearly in screen space
            float inv_depth =
                bary.x * v0.z +
                bary.y * v1.z +
                bary.z * v2.z;

            if (!(inv_depth > 0.0))
                continue;

            // Convert to depth so that smaller means closer.
            float depth = 1.0 / inv_depth;

            // The assignment assumes positive finite float depths.
            // Their uint bit representations preserve numerical order.
            uint depth_bits = asuint(depth);

            uint pixel_index =
                uint(y) * width + uint(x);

            InterlockedMin(
                depth_buffer[pixel_index],
                depth_bits
            );
        }
    }
}


// Rasterize triangle IDs

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
    uint tid : SV_DispatchThreadID,
    uniform uint num_triangles,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> projected_vertices,
    RWStructuredBuffer<uint> depth_buffer,
    RWStructuredBuffer<uint> triangle_id_buffer)
{
    if (tid >= num_triangles)
        return;

    float3 v0, v1, v2;

    if (!get_triangle(tid, faces, projected_vertices, v0, v1, v2))
    {
        return;
    }

    int2 lo, hi;

    if (!get_pixel_bounds(v0, v1, v2, width, height, lo, hi))
    {
        return;
    }

    for (int y = lo.y; y <= hi.y; ++y)
    {
        for (int x = lo.x; x <= hi.x; ++x)
        {
            float2 p = float2(
                float(x) + 0.5,
                float(y) + 0.5
            );

            float3 bary;

            if (!compute_barycentrics(p, v0.xy, v1.xy, v2.xy, bary)){
                continue;
            }

            float inv_depth =
                bary.x * v0.z +
                bary.y * v1.z +
                bary.z * v2.z;

            if (!(inv_depth > 0.0))
                continue;

            float depth = 1.0 / inv_depth;
            uint depth_bits = asuint(depth);

            uint pixel_index =
                uint(y) * width + uint(x);

            // Only triangles at the nearest depth can claim this pixel
            if (depth_bits == depth_buffer[pixel_index])
            {
                InterlockedMin(
                    triangle_id_buffer[pixel_index],
                    tid
                );
            }
        }
    }
}


// Shade pixels

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
    uint2 tid : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform float s,
    uniform float a,
    StructuredBuffer<uint3> faces,
    StructuredBuffer<float3> vertex_colors,
    StructuredBuffer<float3> projected_vertices,
    StructuredBuffer<uint> triangle_id_buffer,
    RWTexture2D<float4> image)
{
    if (tid.x >= width || tid.y >= height)
        return;

    uint pixel_index = tid.y * width + tid.x;

    uint triangle_id = triangle_id_buffer[pixel_index];

    // No triangle covers this pixel. Leave the background color already in the image unchanged.
    if (triangle_id == 0xffffffff)
        return;

    uint3 face = faces[triangle_id];

    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    float2 p = float2(
        float(tid.x) + 0.5,
        float(tid.y) + 0.5
    );

    float3 bary;

    if (!compute_barycentrics(p, v0.xy, v1.xy, v2.xy, bary)){
        return;
    }

    float inv_depth =
        bary.x * v0.z +
        bary.y * v1.z +
        bary.z * v2.z;

    if (!(inv_depth > 0.0))
        return;

    float3 c0 = vertex_colors[face.x];
    float3 c1 = vertex_colors[face.y];
    float3 c2 = vertex_colors[face.z];

    float3 color =
        (
            bary.x * v0.z * c0 +
            bary.y * v1.z * c1 +
            bary.z * v2.z * c2
        ) / inv_depth;

    image[tid] = float4(color, 1.0);
}
