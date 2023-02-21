// BSD 3-Clause License

// Copyright (c) 2021, Muhammad Asad (masadcv@gmail.com)
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <torch/extension.h>
#include <vector>
// #include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

float l1distance(const float &in1, const float &in2)
{
    return std::abs(in1 - in2);
}

// float l2distance(const float *in1, const float *in2, int size)
// {
//     float ret_sum = 0.0;
//     for (int c_i = 0; c_i < size; c_i++)
//     {
//         ret_sum += (in1[c_i] - in2[c_i]) * (in1[c_i] - in2[c_i]);
//     }
//     return std::sqrt(ret_sum);
// }

void euclidean_updown_pass_cpu(
        torch::Tensor &distance
)
{
    // batch, channel, height, width
    const int height = distance.size(2);
    const int width = distance.size(3);

    auto distance_ptr = distance.accessor<float, 4>();
    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    // top-down
    for (int h = 1; h < height; h++)
    {
        // use openmp to parallelize the loop over width
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int w = 0; w < width; w++)
        {
            float cur_dist;
            float new_dist = distance_ptr[0][0][h][w];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int w_ind = w + w_i - 1;

                if (w_ind < 0 || w_ind >= width)
                    continue;

                cur_dist = distance_ptr[0][0][h - 1][w_ind] + local_dist[w_i];
                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }

    // bottom-up
    for (int h = height - 2; h >= 0; h--)
    {
        // use openmp to parallelize the loop over width
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int w = 0; w < width; w++)
        {
            float cur_dist;
            float new_dist = distance_ptr[0][0][h][w];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int w_ind = w + w_i - 1;

                if (w_ind < 0 || w_ind >= width)
                    continue;

                cur_dist = distance_ptr[0][0][h + 1][w_ind] + local_dist[w_i];
                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }
}

torch::Tensor edt2d_cpu(
        const torch::Tensor &mask,
        const int &iterations
)
{
    const int height = mask.size(2);
    const int width = mask.size(3);

    const float dist_init_value = 2.0 * std::sqrt(height * height + width * width);
    torch::Tensor distance = dist_init_value * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        distance = distance.contiguous();

        // top-bottom - width*, height
        euclidean_updown_pass_cpu(distance);

        // left-right - height*, width
        distance = distance.transpose(2, 3);
        distance = distance.contiguous();

        euclidean_updown_pass_cpu(distance);

        // tranpose back to original - width, height
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }

    return distance;
}

void geodesic_updown_pass_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    const float &l_grad,
    const float &l_eucl
    )
{
    // batch, channel, height, width
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    auto image_ptr = image.accessor<float, 4>();
    auto distance_ptr = distance.accessor<float, 4>();
    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    // top-down
    for (int h = 1; h < height; h++)
    {
        // use openmp to parallelise the loop over width
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (int w = 0; w < width; w++)
        {
            float l_dist, cur_dist;
            float new_dist = distance_ptr[0][0][h][w];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int w_ind = w + w_i - 1;

                if (w_ind < 0 || w_ind >= width)
                    continue;

                l_dist = 0.0;
                if (channel == 1)
                {
                    l_dist = l1distance(
                        image_ptr[0][0][h][w],
                        image_ptr[0][0][h - 1][w_ind]
                        );
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        l_dist += l1distance(
                            image_ptr[0][c_i][h][w],
                            image_ptr[0][c_i][h - 1][w_ind]
                            );
                    }
                }
                cur_dist = distance_ptr[0][0][h - 1][w_ind] + \
                            l_eucl * local_dist[w_i] + \
                            l_grad * l_dist;

                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }

    // bottom-up
    for (int h = height - 2; h >= 0; h--)
    {
        // use openmp to parallelise the loop over width
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (int w = 0; w < width; w++)
        {
            float l_dist, cur_dist;
            float new_dist = distance_ptr[0][0][h][w];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int w_ind = w + w_i - 1;

                if (w_ind < 0 || w_ind >= width)
                    continue;

                l_dist = 0;
                if (channel == 1)
                {
                    l_dist = l1distance(
                        image_ptr[0][0][h][w],
                        image_ptr[0][0][h + 1][w_ind]
                        );
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        l_dist += l1distance(
                            image_ptr[0][c_i][h][w],
                            image_ptr[0][c_i][h + 1][w_ind]
                            );
                    }
                }
                cur_dist = distance_ptr[0][0][h + 1][w_ind] + \
                            l_eucl * local_dist[w_i] + \
                            l_grad * l_dist;

                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }
}

torch::Tensor generalised_geodesic2d_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations
    )
{
    torch::Tensor image_local = image.clone();
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image_local = image_local.contiguous();
        distance = distance.contiguous();

        // top-bottom - width*, height
        geodesic_updown_pass_cpu(image_local, distance, l_grad, l_eucl);

        // left-right - height*, width
        image_local = image_local.transpose(2, 3);
        distance = distance.transpose(2, 3);

        image_local = image_local.contiguous();
        distance = distance.contiguous();
        geodesic_updown_pass_cpu(image_local, distance, l_grad, l_eucl);

        // tranpose back to original - width, height
        image_local = image_local.transpose(2, 3);
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }

    return distance;
}
