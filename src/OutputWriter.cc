//============================================================================
// Name        : InputReader.cpp
// Author      : Pablo
// Version     :
// Copyright   : Hecho para Óptima.
// Description : Escribe las salidas al algoritmo
//============================================================================

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <OutputWriter.h>
#include <colormod.h>
#include <dirent.h>
#include <stddef.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;  
void crearDir(string dir){
	Color::Modifier green(Color::FG_GREEN);
	Color::Modifier yellow(Color::FG_YELLOW);
	Color::Modifier def(Color::FG_DEFAULT);
	DIR *dirp = opendir(dir.c_str());
	if (dirp == NULL) {
		char* comando = new char[510];

		sprintf(comando, "mkdir %s", dir.c_str());
		std::system(comando);
		cout << green<<"Se ha creado el directorio de salida en la ruta " << dir
				<<def<< endl;
	} else {
		cout <<yellow<< "Ya existia el directorio de salida en la ruta " << dir
				<<def<< endl;
	}

}

OutputWriter::OutputWriter(InputParser* input, InputReader* mpInputReader,int f0, int f1 ){
	string o = "-o";
	string strOutputPath = input->getCmdOption(o);
	crearDir(strOutputPath);
	string directory = "/"+to_string(f0)+"_"+to_string(f1);
	strOutputPath = strOutputPath+directory;
	crearDir(strOutputPath);
	this->strOutputPath = strOutputPath;
	this->radios = mpInputReader->getRadios();
	this->all_img_names = mpInputReader->getImgNames();
	this->labels = mpInputReader->getLabels();
	this->kps = mpInputReader->getKps();
	this->track_ids = mpInputReader->getTrackIds();
}


void OutputWriter::calcularReproyecciones(MapManager* mpMapManager, InputReader* mpInputReader, bool calcularNormal){
	allKfIds = mpMapManager->GetAllKeyFramesId();
	for (auto i : allKfIds) {
		//Reproyecta en el keyframetodos los puntos del mapa (sean o no observados en el keyframe)
		map<int, cv::Point2f> reprojections =
			mpMapManager->ReproyectAllMapPointsOnKeyFrame(i);
		//genera puntos en un plano normal a una distancia de un cm
		map<int, cv::Point2f> normal_points;
		if(calcularNormal)
			normal_points = mpMapManager->CreatePointsOnNormalPlane(i, 1);
		//obtiene las ubicaciones reales de los puntos observados en el frame
		map<int, cv::Point2f> kpsInput = mpInputReader->GetPoints(i);
		//obtiene los ids de todos los puntos del mapa
		vector<int> tracks = mpInputReader->GetTrackIds(i);
		//almacenará los errores de reproyeccion
		map<int, float> errors;

		
		for (auto const& tuple : reprojections){ //itera sobre las reproyecciones

			int track = tuple.first; //extrae el id del punto
			cv::Point2f rep = tuple.second; //extrae la posición de la reproyeccion
			bool observado = kpsInput.count(track) > 0; //verifica si es un punto observado  o no
			cv::Point2f kp, normal;
			if(observado){
				kp = kpsInput[track]; //obtiene su posición real
				errors[track] = cv::norm(rep - kp); //calcula y almacena el error de reproyeccion
				if(calcularNormal) normal = normal_points[track];//obtiene la posición de su normal
			}
		}
		errors_map[i] = errors;
		rep_map[i] = reprojections;
		if(calcularNormal) normal_points_map[i] = normal_points;
	}
	this->mps = mpMapManager->GetAllMapPoints();
}

void OutputWriter::graficarReproyecciones(InputReader* mpInputReader, bool graficarNoVisibles, bool graficarNormal){
	cv::Scalar color_etiqueta = cv::Scalar(255, 0, 0);
	cv::Scalar color_reproyeccion_visible = cv::Scalar(0, 255, 0);
	cv::Scalar color_reproyeccion_no_visible = cv::Scalar(255, 255, 0);
	cv::Scalar color_normal = cv::Scalar(0, 0, 255);
	for (auto const& tuple_frame : rep_map){
		int frame_id = tuple_frame.first;
		map<int, cv::Point2f> reprojections = tuple_frame.second;
		map<int, cv::Point2f> normal_points;
		if(graficarNormal) normal_points = normal_points_map[frame_id];
		cv::Mat img = cv::imread(mpInputReader->GetImageName(frame_id));
		map<int, cv::Point2f> kpsInput = mpInputReader->GetPoints(frame_id);
		for (auto const& tuple_point : reprojections){
			int track = tuple_point.first; //extrae el id del punto
			cv::Point2f rep = tuple_point.second; //extrae la posición de la reproyeccion
			bool observado = kpsInput.count(track) > 0; //verifica si es un punto observado  o no
			if(!observado && !graficarNoVisibles) continue;
			if(observado){
				//dibuja la reproyeccion
				cv::circle(img, rep, 3, color_reproyeccion_visible, 2);
				cv::putText(img, to_string(track), rep, 0, 0.5, color_reproyeccion_visible);
				//dibuja la etiqueta
				cv::Point2f kp = kpsInput[track];
				cv::circle(img, kp, 2, color_etiqueta, 2);
				cv::putText(img, to_string(track), kp, 0, 0.5, color_etiqueta);
				//dibuja la normal
				if(graficarNormal){
					cv::Point2f normal = normal_points[track];
					cv::circle(img, normal, 2, color_normal, 2);
					cv::putText(img, to_string(track), normal, 0, 0.5,color_normal);
				}
			}else{
				//dibuja la reproyeccion
				cv::circle(img, rep, 3, color_reproyeccion_no_visible, 2);
				cv::putText(img, to_string(track), rep, 0, 0.5, color_reproyeccion_no_visible);
			}
		}
		const string ruta = strOutputPath+"/"+to_string(frame_id)+".png";
		cout<<ruta<<endl;
		cv::imwrite(ruta,img);
	}

}


void OutputWriter::guardarTriangulacion(string filename, bool escala){
	ofstream myfile;
	myfile.open(strOutputPath+filename);
	if (myfile.is_open()) {

		//Escribe encabezados
		myfile << "track_id,frame_id,label,X,Y,Z,img_name,x,y,r,xrep,yrep,error,";
		if(escala) myfile << "x1cm,y1cm";
		myfile << endl;

		/*//Encuentra las coordenadas x,y,z de val_1 y val_2
		cv::Point3d val1_3d;
		cv::Point3d val2_3d;
		for(std::map<int,cv::Point3d>::iterator iter = mps.begin(); iter != mps.end(); ++iter)
		{
			//columna "track_id"
			int track_id =  iter->first;
			//columnas "x","y" y "z"
			cv::Point3d p = iter->second;

			if(labels[track_id] == "val_1"){
				val1_3d=p;
			}

			if(labels[track_id] == "val_2"){
				val2_3d=p;
			}

		}*/

		//Itera sobre los mappoints y frames para generar una fila para cada uno
		for(std::map<int,cv::Point3d>::iterator iter = mps.begin(); iter != mps.end(); ++iter)
		{
			//columna "track_id"
			int track_id =  iter->first;
			//columnas "x","y" y "z"
			cv::Point3d p = iter->second;

			//columnas por frame
			for (auto frame_index : allKfIds){

				myfile << track_id <<","<< frame_index <<","<< labels[track_id];

				myfile << ","<<p.x<< ","<<p.y<< ","<<p.z;

				//verifica si el map point es visible en el frame
				vector<int> tracks_del_frame = track_ids[frame_index];
				int index_del_track_en_el_frame = -1;
				for (int kp_index = 0; kp_index < tracks_del_frame.size(); ++kp_index){
					if(tracks_del_frame[kp_index] == track_id)
						index_del_track_en_el_frame =  kp_index;
				}

				//si es visible...
				if(index_del_track_en_el_frame>-1){
					//columna "img_name_"
					myfile << ","<< all_img_names[frame_index];
					//columnas "x_" e "y_"
					cv::Point2f observacion = kps[frame_index][index_del_track_en_el_frame];
					myfile << ","<<observacion.x<< ","<<observacion.y;
					//columna "r_"
					myfile << ","<< radios[frame_index][track_id];
					/*//columnas "vol_real_" y "vol_rep_"
					myfile << ","
							<< vols_real_map[frame_index][index_del_track_en_el_frame]
							<< ","
							<< vols_rep_map[frame_index][index_del_track_en_el_frame];*/
					//columnas "x_rep_" e "y_rep_"
					cv::Point2f reproyeccion = rep_map[frame_index][track_id];
					myfile << ","<<reproyeccion.x<< ","<<reproyeccion.y;
					//columna "error_"
					if(errors_map[frame_index].count(track_id) > 0){
						myfile << ","<< errors_map[frame_index][track_id];
					} else myfile << ",NULL";
					
					//columnas "x_1cm_" e "y_1cm_"
					if(escala){
						cv::Point2f point1cm = normal_points_map[frame_index][index_del_track_en_el_frame];
						myfile << ","<<point1cm.x<< ","<<point1cm.y;
					}

					/*//Distancia a val1
					if(labels[track_id] == "val_1"){
						float d2val1 = cv::norm(val1_3d - p);
						if(d2val1 != 0){
							myfile << "," << d2val1;
						}else{
							myfile << ",NULL";
						}
					}else{
						myfile << ",NULL";
					}

					//Distancia a val2
					if(labels[track_id] == "val_2"){
						float d2val2 = cv::norm(val2_3d - p);
						if(d2val2 != 0){
							myfile << "," << d2val2;
						}else{
							myfile << ",NULL";
						}
					}else{
						myfile << ",NULL";
					}*/

				}

				//sino...
				else {
					myfile << ",NULL"; //columna "img_name_"
					myfile << ",NULL,NULL"; //columnas "x_" e "y_"
					myfile << ",NULL"; //columna "r_"
					//myfile << ",NULL,NULL";//columnas "vol_real_" y "vol_rep_"
					cv::Point2f reproyeccion = rep_map[frame_index][track_id];
					myfile << ","<<reproyeccion.x<< ","<<reproyeccion.y; //columnas "x_rep_" e "y_rep_"
					myfile << ",NULL"; //columna "error_"
					if(escala) myfile << ",NULL,NULL"; //columnas "x_1cm_" e "y_1cm_"
					//myfile << ",NULL"; //columna "d2val1"
					//myfile << ",NULL"; //columna "d2val2"

				}
				//fin de fila
				myfile << endl;
			}
		}
	}
	myfile.close();
}




/*void OutputWriter::guardarOutliers(map<int, float> outliers_prop){
	ofstream outliers_por_track;
	const string ruta = strOutputPath+"/outliers.txt";
	outliers_por_track.open(ruta);
	outliers_por_track <<"Cantidad de outliers por bundle"<<endl;
	for (auto const& x : outliers_prop){
		int t = x.first; float prop = x.second;
		outliers_por_track<<"bundle "<<t<<": "<<prop<<"%"<<endl;
	}
}*/




