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


TH1F * create_TH1F ( const Str & name, const Str & title = "some_numbers" , const UInt n_bins = 10 , const float x_min = 0 , const float x_max = 19 )
{
	TH1F * h = new TH1F ( name.c_str() , title.c_str() , n_bins , x_min , x_max ) ;
	for ( UInt i (1) ; i <= n_bins ; ++i )
	{
		double d = static_cast<float>(i) ;
		h->SetBinContent ( i , 0.5 + 0.7*d + 0.01*d*d ) ;
		h->SetBinError ( i , 0.4*sqrt(static_cast<float>(d)) ) ;
	}
	return h ;
}


TH1D * create_TH1D ( const Str & name, const Str & title = "some_numbers" , const UInt n_bins = 10 , const float x_min = 0 , const float x_max = 19 )
{
	TH1D * h = new TH1D ( name.c_str() , title.c_str() , n_bins , x_min , x_max ) ;
	for ( UInt i (1) ; i < n_bins ; ++i )
	{
		double d = static_cast<float>(i) ;
		h->SetBinContent ( i , 0.5 + 0.7*d + 0.1*d*d ) ;
		h->SetBinError ( i , 0.4*sqrt(static_cast<float>(d)) ) ;
	}
	return h ;
}


TH2F * create_TH2F ( const Str & name, const Str & title = "some_numbers" , const UInt x_bins = 10 , const float x_min = 0 , const float x_max = 19 , const UInt y_bins = 9 , const float y_min = 3 , const float y_max = 16 )
{
	TH2F * h = new TH2F ( name.c_str() , title.c_str() , x_bins , x_min , x_max , y_bins , y_min , y_max ) ;
	for ( UInt i (1) ; i <= x_bins ; ++i )
	{
		for ( UInt j (1) ; j <= y_bins ; ++j )
		{
			double d = static_cast<float>(i*j) ;
			h->SetBinContent ( i , j , 0.5 + 0.7*d + 0.01*d*d ) ;
			h->SetBinError ( i , j , 0.4*sqrt(static_cast<float>(d)) ) ;
		}
	}
	return h ;
}


TGraph * create_TGraph ( const Str & name, const Str & title = "some_numbers" , const UInt n_bins = 10 , const float x_min = 0 , const float x_max = 19 )
{
	std::vector < double > x, y ;
	double x_inc ( (x_max - x_min) / static_cast<float>(n_bins-1) ) ;
	for ( UInt i (0) ; i < n_bins ; ++i )
	{
		double this_x ( x_min + i*x_inc ) ;
		x.push_back( this_x ) ;
		y.push_back( 9 - 0.2*this_x + 0.4*this_x*this_x ) ;
	}
	TGraph * t = new TGraph ( n_bins , &x[0] , &y[0] ) ;
	t -> SetName ( name.c_str() ) ;
	t -> SetTitle ( title.c_str() ) ;
	return t ;
}


TGraphErrors * create_TGraphErrors ( const Str & name, const Str & title = "some_numbers" , const UInt n_bins = 10 , const float x_min = 0 , const float x_max = 19 )
{
	std::vector < double > x, y, ex, ey ;
	double x_inc ( (x_max - x_min) / static_cast<float>(n_bins-1) ) ;
	for ( UInt i (0) ; i < n_bins ; ++i )
	{
		double this_x ( x_min + i*x_inc ) ;
		x.push_back( this_x ) ;
		ex.push_back( 0.5*x_inc ) ;
		y.push_back( 5 - 0.6*this_x + 0.3*this_x*this_x ) ;
		ey.push_back( 3.*x_inc ) ;
	}
	TGraphErrors * t = new TGraphErrors ( n_bins , &x[0] , &y[0] , &ex[0] , &ey[0] ) ;
	t -> SetName ( name.c_str() ) ;
	t -> SetTitle ( title.c_str() ) ;
	return t ;
}



TGraphAsymmErrors * create_TGraphAsymmErrors ( const Str & name, const Str & title = "some_numbers" , const UInt n_bins = 10 , const float x_min = 0 , const float x_max = 19 )
{
	std::vector < double > x, y, ex_l, ey_l, ex_u, ey_u ;
	double x_inc ( (x_max - x_min) / static_cast<float>(n_bins-1) ) ;
	for ( UInt i (0) ; i < n_bins ; ++i )
	{
		double this_x ( x_min + i*x_inc ) ;
		x.push_back( this_x ) ;
		ex_l.push_back( 0.5*x_inc ) ;
		ex_u.push_back( 0.5*x_inc ) ;
		y.push_back( 5 - 0.6*this_x + 0.3*this_x*this_x ) ;
		ey_l.push_back( 3.*x_inc ) ;
		ey_u.push_back( 1.5*x_inc ) ;
	}
	TGraphAsymmErrors * t = new TGraphAsymmErrors ( n_bins , &x[0] , &y[0] , &ex_l[0] , &ex_u[0] , &ey_l[0] , &ey_u[0] ) ;
	t -> SetName ( name.c_str() ) ;
	t -> SetTitle ( title.c_str() ) ;
	return t ;
}


void make_rootfile ()
{
	std::cout << "Running" << std::endl ;
	TFile * file = new TFile ( "make_rootfile_output.root" , "recreate" ) ;

	std::cout << "Making generic plots" << std::endl ;
	TDirectory * dir1 = file -> mkdir ( "dir1" ) ;
	dir1 -> cd() ;
	TH1F * h1 = create_TH1F ( "hist1_th1f" ) ;

	TDirectory * dir2 = file -> mkdir ( "dir2" ) ;
	dir2 -> cd() ;
	TH1F * h2 = create_TH1F ( "hist2_th1f" , "more_numbers" , 8 , 2 , 7 ) ;

	file->cd() ;
	TH1D * h3 = create_TH1D ( "hist3_th1d" , "doubles" , 20 , 0 , 100 ) ;

	TGraph * t1 = create_TGraph ( "g1_tgraph" , "a_graph" , 20 , 0 , 30 ) ;
	t1 -> Write() ;

	TGraphErrors * t2 = create_TGraphErrors ( "g2_tgrapherrors" , "a_graph_with_symmerrs" , 16 , 9 , 19 ) ;
	t2 -> Write() ;

	TGraphAsymmErrors * t3 = create_TGraphAsymmErrors ( "g3_tgrapherrors" , "a_graph_with_asymmerrs" , 16 , 9 , 19 ) ;
	t3 -> Write() ;

	TH2F * h4 = create_TH2F ( "hist4_th2f" , "hist4_th2f" ) ;

	std::cout << "Writing" << std::endl ;
	file -> Write() ;

	std::cout << "Cleaning up" << std::endl ;
	file -> Close() ;
	delete_and_nullify ( file ) ;

	std::cout << "Closing" << std::endl ;
}