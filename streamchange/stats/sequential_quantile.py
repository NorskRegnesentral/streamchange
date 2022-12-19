
class Quantile:

    def __init__(self, alpha):
        

int Z(double x , double y)
{

  if (x <= y){
    return(1);
  }
  return(0);
    
}

int I(double x , double y , double z)
{

  if ( std::abs(x - y) <= z){
    return(1);
  }
  return(0);
  
}


// update \alpha^th quantile estimator with new observed data point x
std::tuple<double,double,double,double,int> update_quantile( std::tuple<double,double,double,double,int> state , const double& x,const double& alpha)
{
 
  double a = 0.25;
  double zeta = std::get<0>(state);
  double fn = std::get<1>(state);
  double d = std::get<2>(state);
  double d0 = std::get<3>(state);
  // counter
  int i = std::get<4>(state);

  zeta = zeta - ( d/(i+1) ) * ( Z( x , zeta ) - alpha );
  fn = ( 1.0/(i+1) )*( i*fn + I( zeta , x , 1.0/sqrt(i+1) )/(2.0/sqrt(i+1)) );
  d = std::min( 1.0/fn , d0*pow( i+1 , a ) );
     
  int counter = i+1;
  state = std::make_tuple( zeta , fn , d , d0 , counter );
  return(state);
  
}