#include "Extraction/extraction.h"
#include "ClaseLogistica/regressionlogistic.h"
#include <iostream>
/*****************************************************************************
 * Universidad Sergio Arboleda
 * Programa CC-IA
 * Materia: HPC2 - Métricas
 * Autor: Jesus Chaves
 * Fecha: 21/09/2022
 * Tema: Introducción ML
 * Objetivo: Funcion principal para el calculo del modelo de regresion lineal.
 *
 * Requerimientos:
 * 1. - Aplicación o clase para la lectura de ficheros (csv), presente en una
 * clase, para la extracción de los datos, la normalizacion de los datos, en
 * general para la manipulacion de los datos.
 * 2. - Crear una clase para el calculo de la Regresión Logística.
 *****************************************************************************/

#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>
#include <fstream>

/* Principal: Captura los argumentos de entrada:
 * 1. Path donde se encuentra el dataset
 * 2. Separador: Delimitador del dataset (;/,/./ /etc)
 * 3. Header: Si tiene o no cabecera (se elimina si tiene) */
int main(int argc, char *argv[]){
    /* Se crea un objeto del tipo Extraction */
    Extraction extraer(argv[1],argv[2],argv[3]);

    /* Leer los datos del fichero por la funcion ReadCSV() del objeto extraer. */
    std::vector<std::vector<std::string>> DataFrame = extraer.ReadCSV();

    /* Para probar la función EigentoFile, y de esa manera imprimir el fichero de
     * datos, se debe definir el numero de filas y columnas del dataset. Basado
     * en los argumentos de entrada */
    int filas = DataFrame.size() + 1;
    int columnas = DataFrame[0].size();

//    std::cout << "Número de filas: " << filas << std::endl;
//    std::cout << "Número de columnas: " << columnas << std::endl;

    Eigen::MatrixXd MatDataFrame = extraer.CSVtoEigen(DataFrame, filas, columnas);
    /* Imprimir el objeto Matriz DataFrame */
//    std::cout << MatDataFrame << std::endl;

    Eigen::MatrixXd difPromedio = MatDataFrame.rowwise() - extraer.Promedio(MatDataFrame);

    /* Se crea una matrix para almacenar la data normalizada */
    Eigen::MatrixXd dataNormalizada = extraer.Normalizador(MatDataFrame, false);
    /* Se imprimen los datos normalizados */
//    std::cout << dataNormalizada << std::endl;

    /* Se imprime el vector de promedios por columna */
    std::cout << "Promedio: \t" << extraer.Promedio(MatDataFrame) << std::endl;
    std::cout << "Desviación Std: " << extraer.DesvStand(difPromedio) << std::endl;

    /* Una vez normalizados los datos se dividen en grupos de entrenamiento y pruebas:
     * X_train
     * X_train
     * y_test
     * y_test
     * Se tomará para entrenamiento el 80% de los datos. */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> div_data = extraer.TrainTestSplit(dataNormalizada, 0.8);
    Eigen::MatrixXd X_train, y_train, X_test, y_test;

    /* div_data presenta la tupla comprimida con 4 matrices. A continuación se debe
     * descomprimir para las cuatro matrices */
    std::tie(X_train, y_train, X_test, y_test) = div_data;

    /* A continuación se instancia el objeto RegressionLinear */
    RegressionLogistic modelo_lr;

    /* Se ajustan los parámetros */
    int dimension = X_train.cols();
    Eigen::MatrixXd W = Eigen::VectorXd::Zero(dimension);
    double b = 0, lambda = 0.0, learning_rate = 0.01;
    bool bandera = true;
    int num_iteraciones = 10000;

    Eigen::MatrixXd dw;
    double db;
    std::list<double> cost_list;

    std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> optimization = modelo_lr.Optimization(W, b, X_train,
                                                                                                                     y_train, num_iteraciones, learning_rate,
                                                                                                                     lambda, bandera);
    /* Se desempaqueta el conjunto de valores de optimo */
    std::tie(W, b, dw, db, cost_list) = optimization;

    /* Se crean las matrices de predicción, (prueba y entrenamiento) */
    Eigen::MatrixXd y_pred_test = modelo_lr.Prediccion(W, b, X_test);
    Eigen::MatrixXd y_pred_train = modelo_lr.Prediccion(W, b, X_train);

//    std::cout << "test: " << y_pred_test << std::endl;
//    std::cout << "train: " << y_pred_train << std::endl;

    /* A continuación se calcula la metrica de accuracy para evaluar que tan bueno
     * es el modelo */
    auto train_accuracy = 100 - ((y_pred_train - y_train).cwiseAbs().mean() * 100);
    auto test_accuracy = 100 - ((y_pred_test - y_test).cwiseAbs().mean() * 100);

    std::cout << "Accuracy de entrenamiento: " << train_accuracy << std::endl;
    std::cout << "Accuracy de prueba: " << test_accuracy << std::endl;

    /* Se imprime numero de filas y columnas de cada uno de los elementos */
//    std::cout << "Cantidad de registros matriz normalizada: " << dataNormalizada.rows() << std::endl;
//    std::cout << "Cantidad de registros X_train: " << X_train.rows() << std::endl;
//    std::cout << "Cantidad de registros y_train: " << y_train.rows() << std::endl;
//    std::cout << "Cantidad de registros X_test: " << X_test.rows() << std::endl;
//    std::cout << "Cantidad de registros y_test: " << y_test.rows() << std::endl;
//    std::cout << "Cantidad de columnas X_train: " << X_train.cols() << std::endl;
//    std::cout << "Cantidad de columnas y_train: " << y_train.cols() << std::endl;
//    std::cout << "Cantidad de columnas X_test: " << X_test.cols() << std::endl;
//    std::cout << "Cantidad de columnas y_test: " << y_test.cols() << std::endl;


    return EXIT_SUCCESS;

}
