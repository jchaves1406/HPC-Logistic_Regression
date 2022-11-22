#ifndef EXTRACTION_H
#define EXTRACTION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>

/* La clase Extraction se compone de las funciones o métodos para manipular
 * el dataset. Se presentan las funciones para:
 * -- Lectura csv
 * -- Promedios
 * -- Normalizacion de datos
 * -- Desviacion standar
 * La clase recibe como parametros de entrada:
 * -- El dataset (path del dataset)".csv"
 * -- El delimitador (separador entre columnas del dataset)
 * -- Si el dataset tiene o no cabecera (header) */
class Extraction{

        /* Se presenta el constructor para los argumentos de entrada de la clase:
         * nombre_dataset, delimitador, header */
        /* Recibe el nombre del fichero CSV */
        std::string setDatos;
        /* Recibe el separador o delimitador */
        std::string delimitador;
        /*Recibe si tiene cabecera el fichero de datos */
        bool header;

public:
        Extraction(std::string datos, std::string separador, bool head):
        setDatos(datos),
        delimitador(separador),
        header(head) {}

        /* Prototipo de funciones propias de la clase */
        /* Cabecera de funcion ReadCSV */
        std::vector<std::vector<std::string>> ReadCSV();
        /* Cabecera de funcion CSVtoEigen */
        Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> setDatos, int filas, int columnas);
        /* Cabecera de funcion Promedio */
        auto Promedio(Eigen::MatrixXd datos) -> decltype (datos.colwise().mean());
        /* Cabecera de funcion DesvStand */
        auto DesvStand(Eigen::MatrixXd data) -> decltype (((data.array().square().colwise().sum())/(data.rows()-1)).sqrt());
        /* Cabecera de funcion Normalizador */
        Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos,  bool normal_target);
        /* Cabecera de funcion TrainTestSplit */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd dataNorm, float size_train);
        void vector_to_file(std::vector<float> vector_datos, std::string file_name);
        void eigen_to_file(Eigen::MatrixXd matrixData, std::string file_name);
};

#endif // EXTRACTION_H
