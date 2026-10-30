StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;

uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix;
uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;
uniform float3 Ks;
uniform float exponent;

Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;

    float3 position = vertices[vid];
    float3 normal = vertex_normals[vid];
    float2 uv = vertex_uvs[vid];

    float4 world_position = mul(model_matrix, float4(position, 1.0));

    output.position = mul(MVP, float4(position, 1.0));

    output.world_position = world_position.xyz;
    output.world_normal = mul(normal_matrix, normal);

    output.uv = uv;

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {

    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;

    float3 n = normalize(input.world_normal);

    float3 l = normalize(light_dir);
    float3 v = normalize(camera_position - input.world_position);

    float3 ambient = 0.2 * Kd;

    float ndotl = max(0.0, dot(n, l));
    float3 diffuse = Kd * light_color * ndotl;

    float3 specular = float3(0.0, 0.0, 0.0);

    if (dot(n, l) > 0.0) {
        float3 r = reflect(-l, n);
        float rdotv = max(0.0, dot(r, v));

        specular = Ks * light_color * pow(rdotv, exponent);
    }

    float3 color = ambient + diffuse + specular;

    return float4(color, 1.0);
}