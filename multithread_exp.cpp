#include <string>
#include <iostream>
#include <thread>

using namespace std;

int task1_answer;
int task2_answer;

void task1(string msg)
{
	volatile unsigned long long i;
	for(i = 0; i < 100000000ULL; ++i);
	task1_answer = 2300;
}

void task2(string msg)
{
	volatile unsigned long long i;
	for(i = 0; i < 100000ULL; ++i);
	task2_answer = 1000;
}

int main()
{
	thread t1(task1, "helloo");
	thread t2(task2, "ahola");
	
	cout << "before joining the answer is " << task1_answer + task2_answer << endl;
	
	t2.join();
	cout << "after joining 1 the answer is " << task1_answer + task2_answer << endl;
	t1.join();
	
	cout << "after joining the answer is " <<    + task2_answer << endl;

}
