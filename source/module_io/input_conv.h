//==========================================================
// Author: Lixin He,mohan
// DATE : 2008-12-24
//==========================================================
#ifndef INPUT_CONVERT_H
#define INPUT_CONVERT_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdio.h>
#include <string.h>
#include <regex.h>
#include <vector>

using namespace std;


namespace Input_Conv
{
    void Convert(void);

    // Function `parse_expression` is used to parse input parameters as expressions into vectors
    // fn (string): expressions such as "3*1 0 2*0.5 3*0"
    // arr (vector): stores parsing results, for example, "3*1 0 2*0.5 1*1.5" can be parsed as [1, 1, 1, 0, 0.5, 0.5, 1.5]
    void parse_expression(const std::string &fn, std::vector<double> &arr);
}


#endif //Input_Convert