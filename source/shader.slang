struct vertex_input_t
{
	float3 position;
	float3 normal;
	float2 uv;
};

Sampler2D textures[];

struct shader_data_t
{
	float4x4 projection;
	float4x4 view;
	float4x4 model[3];
	float4 light_position;
	uint32_t selected;
};

struct vertex_output_t
{
	float4 position : SV_POSITION;
	float3 normal;
	float2 uv;
	float3 factor;
	float3 light_vector;
	float3 view_vector;
	uint32_t instance_index;
};

[shader ("vertex")]
vertex_output_t main (vertex_input_t input, uniform shader_data_t* shader_data, uint instance_index : SV_VulkanInstanceID)
{
	vertex_output_t output;
	float4x4 model_matrix = shader_data->model[instance_index];
	output.normal = mul((float3x3) mul (shader_data->view, model_matrix), input.normal);
	output.uv = input.uv;
	output.position = mul (shader_data->projection, mul (shader_data->view, mul (model_matrix, float4 (input.position.xyz, 1.0))));
	output.factor = (shader_data->selected == instance_index ? 3.0f : 1.0f);
	output.instance_index = instance_index;

	// calculate view vectors required for lighting
	float4 frag_position = mul (mul (shader_data->view, model_matrix), float4 (input.position.xyz, 1.0));
	output.light_vector = shader_data->light_position.xyz - frag_position.xyz;
	output.view_vector = -frag_position.xyz;

	return output;
}

[shader ("fragment")]
float4 main (vertex_output_t input)
{
	// phong lighting
	float3 normal = normalize (input.normal);
	float3 light = normalize (input.light_vector);
	float3 view = normalize (input.view_vector);
	float3 ref = reflect (-light, normal);
	float3 diffuse = max (dot (normal, light), 0.0025);
	float3 specular = pow (max (dot (ref, view), 0.0), 16.0) * 0.75;

	// sample from texture
	float3 color = textures[NonUniformResourceIndex (input.instance_index)].Sample (input.uv).rgb * input.factor;
	return float4 (diffuse * color.rgb + specular, 1.0);
}
