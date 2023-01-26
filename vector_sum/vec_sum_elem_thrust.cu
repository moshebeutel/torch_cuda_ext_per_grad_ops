#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../utils/utils.h"

int main(void)
{
    // initialize data at host side
    auto iStart = time_of_day_seconds();
    
    // generate 32M random numbers serially
    thrust::host_vector<int> h_x(32 << 20);
    std::generate(h_x.begin(), h_x.end(), rand);
    auto iElapsInitData = time_of_day_seconds() - iStart;
    
    // transfer data to the device
    thrust::device_vector<int> d_x = h_x;
    auto iElapsTransferToDevice = time_of_day_seconds() - iElapsInitData;

    // compute sum
    auto sum = thrust::reduce(d_x.begin(), d_x.end());
    auto iElapsSumOnDevice = time_of_day_seconds() - iElapsTransferToDevice;
    
    
    std::cout << "SUM: " << sum << std::endl;
    std::cout << "initialData Time elapsed " << iElapsInitData << "sec." << std::endl
              << "Transfer Data To Device elapsed " << iElapsTransferToDevice << " sec." << std::endl
              << "Sum elapsed " << iElapsSumOnDevice << " sec." << std::endl;


    return 0;
}

