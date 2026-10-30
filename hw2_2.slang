//helper to compute one edge
float edge_value(float2 vi, float2 vj, float2 p) {      //E_(ij)(p) = (p - v_(i)) * n_(ij)
    float2 e = vj - vi;
    float2 n = float2(e.y, -e.x);
    return dot(p - vi, n);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1
    
    float2 p = float2(ipixel) + 0.5;
    float e01 = edge_value(sp0, sp1, p);
    float e12 = edge_value(sp1, sp2, p);
    float e20 = edge_value(sp2, sp0, p);

    float sum = e01 + e12 + e20; // calculate 2d barycentric coords, edge function value proportional to area
    if (abs(sum) < 1e-6) { //   Degenerated triangles check
        return;
    }                   
    float bp0 = e12 / sum; // edge opposite vertex 0
    float bp1 = e20 / sum; // edge opposite vertex 1
    float bp2 = e01 / sum; // edge opposite vertex 2

    float denom = bp0 * inv_depth0 + bp1 * inv_depth1 + bp2 * inv_depth2; //3d barycentric coords
    float b0 = bp0 * inv_depth0 / denom;
    float b1 = bp1 * inv_depth1 / denom;
    float b2 = bp2 * inv_depth2 / denom;

    float depth = b0 / inv_depth0 + b1 / inv_depth1 + b2 / inv_depth2;  //depth p.z = b0p0.z + b1p1.z + b2p2.z

    bool all_nonneg = (e01 >= 0 && e12 >= 0 && e20 >= 0);
    bool all_nonpos = (e01 <= 0 && e12 <= 0 && e20 <= 0);

    if (all_nonneg || all_nonpos) {
        if (depth < depth_image[ipixel]) { // 3D barycentrics + depth, checks if pixel is closer to camera
            depth_image[ipixel] = depth;
            float3 c = b0 * c0 + b1 * c1 + b2 * c2;
            color_image[ipixel] = float4(c, 1.0);
        }
        
    }
}