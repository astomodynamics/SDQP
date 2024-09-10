#include <iostream>

#include "sdqp/sdqp.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    int n = 2;
    int m = 6;
    Eigen::MatrixXd Q(n, n);
    Eigen::MatrixXd c(n, 1);
    Eigen::Matrix<double, 2, 1> x;         // decision variables
    Eigen::MatrixXd A(m, n); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    Q << 4.0, 1.0,
         0.0, 2.0;
    c << 1.0, 1.0;

    Eigen::MatrixXd A_dense(m/2, n);
    A_dense << 1.0, 1.0,
               1.0, 0.0,
               0.0, 1.0;

    // A matrix for both lower and upper bounds
    A << A_dense, -A_dense;
    
    // lower and upper bounds
    Eigen::Vector3d l, u;
    l << 1.0, 0.0, 0.0;
    u << 1.0, 0.7, 0.7;
    
    // combine lower and upper bounds
    b << u, -l;

    double minobj = sdqp::sdqp<2>(2*Q, c, A, b, x);

    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << minobj << std::endl;
    std::cout << "cons precision: " << (A * x - b).maxCoeff() << std::endl;

    return 0;
}
