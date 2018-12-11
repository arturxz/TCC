# include <stdio.h>
# include <stdlib.h>

int func1(int x, int y){
	return ( x & y || (x==0 && y==0));
}

int func2(int x, int y){
	int ctx = x & y;
	if(ctx > 0){
		return 1;
	} else if( (x == 0) && (y == 0) ){
		return 1;
	} else{
		return 0;
	}
}

void compara(int x, int y){
	if( func1(x, y) == func2(x, y) ){
		printf("Equivalentes para entrada x=%i e y=%i.\n", x, y);
	} else{
		printf("-> ERRO PARA ENTRADA x=%i e y=%i.\n", x, y);
		printf("--> Saida func1: %i\n", func1(x, y));
		printf("--> Saida func2: %i\n", func2(x, y));
	}
}

int main(){
	
	compara(0, 0);
	compara(1, 0);
	compara(0, 1);
	compara(1, 1);
	compara(1, 2);
	compara(2, 1);
	compara(2, 2);
	compara(2, 0);
	compara(0, 2);
	
	return 0;
}
