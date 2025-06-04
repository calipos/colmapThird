// test.cpp : Defines the entry point for the application.
//
#include <map>
#include <vector>
#include <random>
#include <string> 
#include <iostream>
#define EACH_ROW 11
void show(int a, int b)
{
	for (int i = 0; i < EACH_ROW; i++)
	{
		if (i == a || i == b)
		{
			std::cout << 1;
		}
		else
		{
			std::cout << ".";
		}
	}
	std::cout << std::endl;
}
int main()
{
	for (int i = 1; i < 20; i++)
	{
		int row = std::ceil(0.5*(sqrt(8*i+1)-1))  ;
		 
		int col = row * (row + 1) / 2-i;
		show(row, col);
	}
	return 0;
	{
		std::vector<float>data;
		std::vector<int>dataInt;
		data.resize(128);
		dataInt.resize(128);
		for (size_t i = 0; i < 128; i++)
		{
			dataInt[i] = 4 * i;
			data[i] = dataInt[i]/512.;
		}
	
	}
	std::vector<float>data;
	std::vector<int>dataInt;
	data.resize(128);
	dataInt.resize(128);
	float sum = 0;
	for (int i = 0; i < 128; i++)
	{
		data[i] = 1+0.5*i;
		sum += (data[i] * data[i]);
	}
	sum = 1. / sqrt(sum);
	for (int i = 0; i < 128; i++)
	{
		data[i] *= sum; 
		dataInt[i] = 512*data[i];
	}
	return 0;
}
