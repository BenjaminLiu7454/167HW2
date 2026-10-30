StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_colors;
uniform float4x4 MVP;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;

    float3 vertex = vertices[vid];

    // Transform the vertex from object space to clip space
    output.position = mul(MVP, float4(vertex, 1.0));

    // Pass the vertex color to the fragment shader
    output.color = vertex_colors[vid];

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // Output the interpolated RGB color with full opacity
    return float4(input.color, 1.0);
}