/*
 * Filter.h
 *
 *  Created on: Jan 15, 2011
 *      Author: Alexander Shishkov
 */

#ifndef FILTER_H_
#define FILTER_H_

#include <vector>

#include <tod/detecting/GuessGenerator.h>
#include <tod/core/TexturedObject.h>
#include <opencv2/core/core.hpp>

namespace tod
{
  class Filter
  {
  public:
    virtual ~Filter() {};
    virtual void filterGuesses(std::vector<Guess>& guesses) = 0;
  };

  class StdDevFilter: public Filter
  {
  public:
    StdDevFilter(const GuessGeneratorParameters& generatorParams);
    ~StdDevFilter() {};
    void filterGuesses(std::vector<Guess>& guesses);
  private:
    const GuessGeneratorParameters& params;
  };

  class OverlappingFilter: public Filter
  {
  public:
    OverlappingFilter() {};
    ~OverlappingFilter() {};
    void filterGuesses(std::vector<Guess>& guesses);
  };

  class CloudFilter: public Filter
  {
  public:
    CloudFilter(int width, int height);
    ~CloudFilter() {};
    void filterGuesses(std::vector<Guess>& guesses);
  private:
    int width;
    int height;
  };
}
#endif /* FILTER_H_ */


