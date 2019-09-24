/** Erosion of src image by mask. Result is stored in dst image.

  */

// ARRAY convolution_window [window_size_x*window_size_y]
// SCALAR window_size_x
// SCALAR window_size_y

__kernel void vglClDilate(__read_only image2d_t img_input,
                          __write_only image2d_t img_output,
                          __constant float* convolution_window, 
                          int window_size_x, 
                          int window_size_y)
{
    int2 coords = (int2)(get_global_id(0), get_global_id(1));
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_CLAMP_TO_EDGE |   //Clamp to next edge
                          CLK_FILTER_NEAREST;           //Don't interpolate
    
    int factorx = floor((float)window_size_x / 2.0f);
    int factory = floor((float)window_size_y / 2.0f);
    int conv_controller = 0;
    float4 pmax = 0.0f;
    for(int i = -factory; i <= factory; i++)
    {
        for(int j = -factorx; j <= factorx; j++)
        {
            float4 p = read_imagef(img_input, smp, (int2)(coords.x + j, coords.y + i));
            if (!(convolution_window[conv_controller] == 0))
                pmax = max(p, pmax);
            conv_controller++;
        }
    }
    write_imagef(img_output, coords, pmax);
}
