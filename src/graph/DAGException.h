#ifndef DAGEXCEPTION_H
#define DAGEXCEPTION_H

#include <iostream>
#include <exception>

using namespace std;

struct DAGException : public exception
{
	const char * what () const throw ()
    {
    	return "\nThe computation graph is not Directed and acyclic";
    }
};

#endif
