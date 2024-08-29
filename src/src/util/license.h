////////////////////////////////////////////////////////////////////////////////
// This file is part of RRCA, the Roadrunner Covariance Analsys package.      //
//                                                                            //
// Copyright (c) 2021, Michael Multerer and Paul Schneider                    //
//                                                                            //
// All rights reserved.                                                       //
//                                                                            //
// This source code is subject to the BSD 3-clause license and without        //
// any warranty, see <https://github.com/muchip/RRCA> for further             //
// information.                                                               //
////////////////////////////////////////////////////////////////////////////////
#ifndef RRCA_LICENSE_LICENSEGENERATOR_H_
#define RRCA_LICENSE_LICENSEGENERATOR_H_

#include <array>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace RRCA {
namespace License {

time_t read_timestamp(const std::vector<size_t> &enc_timestamp,
                      size_t offset = 0) {
  std::array<size_t, 40> mult = {1, 2, 3, 6, 5, 2, 3, 5, 1, 5, 4, 2, 3, 5,
                                 8, 4, 7, 4, 6, 9, 1, 2, 3, 6, 3, 2, 3, 5,
                                 1, 5, 4, 2, 3, 5, 2, 4, 7, 4, 6, 2};
  time_t rec_date = size_t(enc_timestamp[0] / 1000) / mult[0 + offset] - 48;
  for (auto i = 1; i < enc_timestamp.size(); ++i)
    rec_date =
        rec_date * 10 + size_t(enc_timestamp[i] / 1000) / mult[i + offset] - 48;
  return rec_date;
}

std::vector<size_t> write_timestamp(time_t timestamp, size_t offset = 0) {
  std::vector<size_t> enc_time;

  std::array<size_t, 40> mult = {1, 2, 3, 6, 5, 2, 3, 5, 1, 5, 4, 2, 3, 5,
                                 8, 4, 7, 4, 6, 9, 1, 2, 3, 6, 3, 2, 3, 5,
                                 1, 5, 4, 2, 3, 5, 2, 4, 7, 4, 6, 2};
  unsigned int i = 0;
  std::string bla = std::to_string(timestamp);
  for (auto &&d : bla) {
    enc_time.push_back(d * 1000 * mult[i + offset] + (rand() % 1000));
    ++i;
  }
  return enc_time;
}

void write_license_file(unsigned int days_till_expiry,
                        const std::string &fname) {
  time_t cur_time = time(0);
  time_t exp_time = cur_time + 24 * 3600 * days_till_expiry;
  std::string exp_string(std::ctime(&exp_time));
  std::ofstream myfile;
  myfile.open(fname);
  myfile << "Trial license for the RRCA package. Valid until " << exp_string;
  srand(cur_time);
  {
    std::vector<size_t> cur_time_stmp = write_timestamp(cur_time, 4);
    for (auto &&i : cur_time_stmp)
      myfile << i << " ";
    myfile << std::endl;
  }
  {
    std::vector<size_t> cur_time_stmp = write_timestamp(exp_time, 2);
    for (auto &&i : cur_time_stmp)
      myfile << i << " ";
    myfile << std::endl;
  }
  {
    std::vector<size_t> cur_time_stmp = write_timestamp(cur_time, 3);
    for (auto &&i : cur_time_stmp)
      myfile << i << " ";
    myfile << std::endl;
  }
  myfile.close();
}

bool check_license_file(const std::string &fname) {
  time_t cur_time = time(0);
  time_t last_time;
  time_t exp_time;
  std::string firstLine;
  std::string secLine;
  std::string thirdLine;
  std::string fourthLine;
  {
    std::ifstream myfile;
    // read license file
    myfile.open(fname);
    if (!myfile.is_open()) {
      std::cerr << "error in license file" << std::endl;
      return false;
    }
    if (!getline(myfile, firstLine))
      std::cerr << "error in license file";
    if (!getline(myfile, secLine))
      std::cerr << "error in license file";
    if (!getline(myfile, thirdLine))
      std::cerr << "error in license file";
    if (!getline(myfile, fourthLine))
      std::cerr << "error in license file";
    myfile.close();
  }
  {
    size_t num = 0;
    std::vector<size_t> enc_exp_time;
    std::stringstream stream(thirdLine);
    while (stream >> num)
      enc_exp_time.push_back(num);
    exp_time = read_timestamp(enc_exp_time, 2);
  }
  {
    size_t num = 0;
    std::vector<size_t> enc_last_time;
    std::stringstream stream(fourthLine);
    while (stream >> num)
      enc_last_time.push_back(num);
    last_time = read_timestamp(enc_last_time, 3);
  }
  if (cur_time < last_time || cur_time > exp_time) {
    std::cout << "This license expired on " << std::ctime(&exp_time);
    return false;
  } else {
//     std::cout << "This license expires on " << std::ctime(&exp_time);
//     std::cout << "-->" << int((exp_time - cur_time) / 3600 / 24)
//               << " day(s) of trial left" << std::endl;
    std::ofstream myfile;
    // read license file
    myfile.open(fname);
    if (!myfile.is_open()) {
      std::cerr << "error in license file" << std::endl;
      return false;
    }
    myfile << firstLine << std::endl
           << secLine << std::endl
           << thirdLine << std::endl;
    // rewrite last line
    std::vector<size_t> cur_time_stmp = write_timestamp(cur_time, 3);
    for (auto &&i : cur_time_stmp)
      myfile << i << " ";
    myfile << std::endl;
    myfile.close();
    return true;
  }
}

} // namespace License

} // namespace RRCA

#endif
