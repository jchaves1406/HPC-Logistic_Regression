#include "regressionlogistic.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <list>

/* Primera función  miembro: La función Sigmoid */
Eigen::MatrixXd RegressionLogistic::Sigmoid(Eigen::MatrixXd Z){
    /* Función Sigmoid que retorna la matriz con el cálculo de la función */
    return 1/(1+(-Z.array()).exp());
}

/* Segunda función: La función de propagación la cual contiene el tratamiento de la función
 * de costo (Cross Entropy), y sus corrrespondientes derivadas */
std::tuple<Eigen::MatrixXd, double, double> RegressionLogistic::Propagation(Eigen::MatrixXd W, Eigen::MatrixXd X, double b,
                                                                            Eigen::MatrixXd y, double lambda){
    /* Sobre la base de la presentación de Regresión Logistica */
    int m = y.rows();

    /* Se obtiene la matriz Eigen Z */
    Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array()+b;
    Eigen::MatrixXd A = Sigmoid(Z);

    /* Se crea una función para la entropia cruzada:
     * No sabemos que valor debera retornar */
    auto cross_entropy = -(y.transpose()*(Eigen::VectorXd)A.array().log().transpose()+((Eigen::VectorXd)(1-y.array())).transpose()*(Eigen::VectorXd)(1-A.array()).log().transpose())/m;

    /* Función para reducir la varianza del modelo usando la regularización */
    double L2_regulation_cost = W.array().pow(2).sum()*(lambda/(2*m));

    /* Función para el cálculo del costo usando la entropia cruzada con el ajuste
     * de regularización:
     * Se hace uso del static_cast debido a que la funcion debe retornar un double
     * pero va a estar compuesta de tipos de datos definidos por el usuario (auto). */
    double cost = static_cast<const double> (cross_entropy.array()[0] + L2_regulation_cost);

    /* Se calculan las derivadas de las matrices en función de los pesos */
    Eigen::MatrixXd dw = ((Eigen::MatrixXd)(A-y.transpose())*X/m) + (Eigen::MatrixXd)(lambda/m*W).transpose();

    /* Se calcula la derivada en funcion del bias (punto de corte) */
    double db = (A-y.transpose()).array().sum()/m;

    /* Se retorna de la funcion propagación la derivada de los pesos dw, derivada
     * de bias db y el costo cost; El retorno es una tupla comprimida */
    return std::make_tuple(dw, db, cost);
}

/* Tercera función: Se crea la función de optimización:
 * Se crea una lista en donde se va a almacenar cada uno de los valores de la función
 * de costo hasta converger. Esta actualización se allacenará en un ficero para posteriormente
 * ser visualizada. La actualización se ve representada la presentación de Regresión
 * Logística. Se pasa un bandera a la función para imprimir si se quiere el valor del costo
 * cada 100 iteraciones */
std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> RegressionLogistic::Optimization(Eigen::MatrixXd W, double b, Eigen::MatrixXd X,
                                                                                                                 Eigen::MatrixXd y, int n_iter, double learning_rate,
                                                                                                                 double lambda, bool log_cost){
    /* Se crea la lista */
    std::list<double> list_cost;
    Eigen::MatrixXd dw;
    double db, cost;

    /* Se hace la iteracion */
    for(int i = 0; i < n_iter; i++){
        std::tuple<Eigen::MatrixXd, double, double> propagation = Propagation(W, X, b, y, lambda);
        std::tie(dw, db, cost) = propagation;

        /* Se actualizan los valores (W, b), que representan los weights y biases */
        W = W - (learning_rate * dw).transpose();
        b = b - (learning_rate * db);

        /* Según la bandera se guarda cada 100 pasos el valor del costo */
        if(i % 100 == 0){
            list_cost.push_back(cost);
        }

        /* Se imprime por pantalla segun la bandera */
        if(log_cost & i % 100 == 0){
            std::cout << "Costo despues de iteracion " << i << ": " << cost << std::endl;
        }
    }

    return std::make_tuple(W, b, dw, db, list_cost);
}

/* Función de predicción: La función estimará (predicción) la etiqueta de salida si
 * corresponde a 0 o 1, la idea es calcular ŷ (y estimado) usando los parámetros de
 * regresión (W, b) aprendidos.
 * Se convierten las entradas a 0 si la función de activación es inferior o igual a 0.5
 * se convierten las entradas a 1 si la función de activación es superior a 0.5. */
Eigen::MatrixXd RegressionLogistic::Prediccion(Eigen::MatrixXd W, double b, Eigen::MatrixXd X){
    /* Se calculan la cantidad de valores de registros (m) */
    int m = X.rows();

    /* Se crea una matriz con valores del vector de zeros del tamaño de la matriz
     * de entrada (X) */
    Eigen::MatrixXd y_predict = Eigen::VectorXd::Zero(m).transpose();

    /* Se crea una matriz para almacenar los valores de Z (calculados) */
    Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array() + b;

    /* Se calcula la funcion sigmoid en la matriz A */
    Eigen::MatrixXd A = Sigmoid(Z);

    /* Se cálcula el valor estimado (Etiquetas 0, 1), según la función de activación
     * cada uno de los registros(matriz X) */
   for(int i = 0; i < A.cols(); i++){
       if(A(0, i) <= 0.5){
           y_predict(0, i) = 0;
       }else{
           y_predict(0, i) = 1;
       }
   }

   return y_predict.transpose();
}







































