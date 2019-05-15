// ====================================================================================================
//  Brief: create a rootfile to ply with
//  Author: Stephen Menary (stmenary@cern.ch)
// ====================================================================================================


using UInt = unsigned int ;
using Str = std::string ;

#include <iostream>
#include <memory>


template<class T> void delete_and_nullify ( T * & ptr )
{
	delete ptr ;
	ptr = NULL ;
}


TH2F * create_TH2F ( const Str & name, const Str & title = "some_numbers" , const UInt x_bins = 10 , const float x_min = 0 , const float x_max = 19 , const UInt y_bins = 9 , const float y_min = 3 , const float y_max = 16 , const int & rand_seed = 100 )
{
	TH2F * h = new TH2F ( name.c_str() , title.c_str() , x_bins , x_min , x_max , y_bins , y_min , y_max ) ;
	TRandom3 * rand = new TRandom3(rand_seed) ;
	for ( UInt i (1) ; i <= x_bins ; ++i )
	{
		for ( UInt j (1) ; j <= y_bins ; ++j )
		{
			double d = static_cast<float>(i*j) ;
			h->SetBinContent ( i , j , ( 0.5 + 0.7*d + 0.01*d*d )*(0.5*rand->Rndm()+0.75) ) ;
			h->SetBinError ( i , j , 0.4*sqrt(static_cast<float>(d))*(rand->Rndm()+1.) ) ;
		}
	}
	delete_and_nullify(rand) ;
	return h ;
}


void make_rootfile ()
{
	std::cout << "Running" << std::endl ;
	TFile * file = new TFile ( "make_rootfile_output.root" , "recreate" ) ;

	std::cout << "Making generic plots" << std::endl ;

	TH2F * h4 = create_TH2F ( "hist1_th2f numerator" , "hist1_th2f numerator" , 3 , 0 , 3 , 2 , 1 , 4 , 3524543) ;

	TH2F * h5 = create_TH2F ( "hist2_th2f denominator" , "hist2_th2f denominator" , 3 , 0 , 3 , 2 , 1 , 4 , 90238462 ) ;

	std::cout << "Writing" << std::endl ;
	file -> Write() ;

	std::cout << "Cleaning up" << std::endl ;
	file -> Close() ;
	delete_and_nullify ( file ) ;

	std::cout << "Closing" << std::endl ;
}