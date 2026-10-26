//helper to compute one edge
float edge_value(float2 vi, float2 vj, float2 p) {      //E_(ij)(p) = (p - v_(i)) * n_(ij)
    float2 e = vj - vi;
    float2 n = float2(e.y, -e.x);
    return dot(p - vi, n);
}

[shader("compute")]
[numthreads(8, 8, 1)]

void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1
    float2 p = float2(ipixel) + 0.5;
    float e01 = edge_value(sp0, sp1, p);
    float e12 = edge_value(sp1, sp2, p);
    float e20 = edge_value(sp2, sp0, p);

    bool all_nonneg = (e01 >= 0 && e12 >= 0 && e20 >= 0);
    bool all_nonpos = (e01 <= 0 && e12 <= 0 && e20 <= 0);
    
    if (all_nonneg || all_nonpos) {
        image[ipixel] = float4(color, 1.0);
    }
}
