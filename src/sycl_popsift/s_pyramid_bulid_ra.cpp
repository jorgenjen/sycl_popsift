#include "sycl_popsift/sift_pyramid.hpp"

namespace popsift {

namespace normalizedSource {

// NOTE: Should probably be writen as a functor instead of a lambda inside of a function
// swap out the texture memory with normal memory or work-group memory (scratch pad memory/shared memory)
// and then the rest of the kernel should be the same just use nd_range and it should be the same as this
// quite straight forward (I think and HOPE!)

// __global__ static void horiz(
static void horiz(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_w, int dst_h, float shift)
{
    // Create octave-0 - level-0 from the input image.
    const int write_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int write_y = blockIdx.y;

    if(write_x >= dst_w)
        return;

    const int span = d_gauss.dd.span[0];
    const float* filter = &d_gauss.dd.filter[0];
    const float read_x = (blockIdx.x * blockDim.x + threadIdx.x + shift) / dst_w;
    const float read_y = (blockIdx.y + shift) / dst_h;

    if(write_y == 10 && write_x == 10)
    {
        printf("span: %d\n", span);
        printf("filter: %f\n", filter[0]);
        printf("read_x: %f\n", read_x);
        printf("read_y: %f\n", read_y);

        // for(int i = 0; i >
    }
    // std::cout << "span for horiz: " << span << std::endl;
    // std::cout << "filter for horiz: " << filter << std::endl;

    float out = 0.0f;

#pragma unroll
    for(int offset = span; offset > 0; offset--)
    {
        const float& g = filter[offset];
        const float offrel = float(offset) / dst_w;
        const float v1 = tex2D<float>(src_linear_tex, read_x - offrel, read_y);
        const float v2 = tex2D<float>(src_linear_tex, read_x + offrel, read_y);
        out += ((v1 + v2) * g);
        if(write_y == 10 && write_x == 10)
        {
            printf("\n\n");
            printf("offset: %d\n", offset);
            printf("g: %f\n", g);
            printf("offrel: %f\n", offrel);
            printf("v1: %f\n", v1);
            printf("v2: %f\n", v2);
            printf("out: %f\n", out);
            printf("\n\n");
        }
    }
    const float& g = filter[0];
    const float v3 = tex2D<float>(src_linear_tex, read_x, read_y);
    out += (v3 * g);

    surf2DLayeredwrite(out * 255.0f, dst_data, write_x * 4, write_y, 0, cudaBoundaryModeZero);
}

} // namespace normalizedSource

__host__ void Pyramid::horiz_from_input_image(const Config& conf, ImageBase* base, cudaStream_t stream)
{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block(128, 1);
    dim3 grid;
    grid.x = grid_divide(width, 128);
    grid.y = height;

    float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

    std::cout << "grid: " << grid.x << " " << grid.y << std::endl;
    std::cout << "shift: " << shift << std::endl;
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    // std::cout << "Gaussion Span: " << d_gauss.dd.span[0] << std::endl;
    // std::cout << "Gaussian Filter: " << d_gauss.dd.filter[0] << std::endl;

    // float* vec_start = &d_gauss.dd.filter[0];
    // for( int i=0; i < 128; i++){
    //   std::cout << "Gaussian Filter[" << i << "]: " << vec_start[i] << std::endl;
    // }

    // const int    span    =  d_gauss.dd.span[0];
    // const float* filter  = &d_gauss.dd.filter[0];

    normalizedSource::horiz<<<grid, block, 0, stream>>>(
      base->getInputTexture(), oct_obj.getIntermediateSurface(), width, height, shift);

    POP_SYNC_CHK;
}

} // namespace popsift
