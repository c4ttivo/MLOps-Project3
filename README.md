![alt text](https://github.com/c4ttivo/MLOpsTaller1/blob/main/mlopstaller1/imgs/logo.png?raw=true)

# MLOps - Proyecto 3
## Autores
*    Daniel Crovo (dcrovo@javeriana.edu.co)
*    Carlos Trujillo (ca.trujillo@javeriana.edu.co)

## Profesor
*    Cristian Diaz (diaz.cristian@javeriana.edu.co)

## Instalación Minikube

Para publicar los servicios en kubernetes es necesario usar minikube, que habilita la máquina virtual para desplegar servicios de kubernetes. 

### Prerequisitos

*    Sistema operativo Ubuntu 22.04
*    2GB o más de RAM
*    2 CPU o más
*    20 GB de espacio libre en disco
*    Privilegios de administrador con sudo
*    Conexión a internet
*    Docker

Los pasos a seguir para la instalación se presentan a continuación.

### Actualizar el sistema

```
$ sudo apt update
$ sudo apt upgrade -y
```

Los anteriores comandos actualizan el repositorio y el sistema operativo antes de la instalación de minikube. Después de instalar las actualizaciones es requerido reiniciar el sistema.

```
$ sudo reboot
```

### Ejecutar docker sin requerir sudo

```
$ sudo usermod -aG docker $USER
$ newgrp docker
```

Para que los cambios tomen efecto se debe realizar logout y login.

### Descargar e instalar Minikube

```
$ curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
$ sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

Para verificar, se ejecuta el siguiente comando:

```
$ minikube version
```

### Instalar herramienta kubectl

```
$ curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
```

Luego se establecen permisos de ejecución y se mueve al directorio /usr/local/bin/minikube

```
$ chmod +x kubectl
$ sudo mv kubectl /usr/local/bin/
```

Para verificar la versión de kubectl, se ejecuta:

```
$  kubectl version -o yaml
```

### Iniciar cluster de Minikube

```
$ minikube start --driver=docker -p puj
```

Una vez se ha iniciado el cluster, para ejecutar el estado del cluster, se ejecuta:

```
$ minikube -p puj status
```

### Interactuar con el cluster de Minikube

A través de kubectl se interactua con el cluster.

```
$ kubectl get nodes
$ kubectl cluster-info
```

## Arquitectura

![alt text](https://github.com/c4ttivo/MLOPS-Project3/blob/main/img/architecture.png?raw=true)

### Descripción de componentes

La solución está compuesta por los siguientes contenedores:

*   **Docker Compose**:
	*	**Airflow**: Orquestador para gestionar y programar los flujos de trabajo, relacionados con la recolección de datos, entrenamiento de modelos y registro de experimentos en MLFlow.
	*	**MLflow**: Registro de experimentos, seguimiento de métricas y almacenamiento de modelos. Configurado para usar **Minio** para el almacenamiento de objetos y **MySQL** como base de datos para la metadata.
	*	**Minio**: Almacen de objetos compatible con S3.
	*	**Postgresql**: Se encuentran dos servicios, uno de apoyo a MLflow y otro para el almacenamiento de la recolección de datos (modeldb).
*   **Kubernetes**:
	*	**Inference**: Servicio de FastAPI que consume el modelo entrenado y almacenado en MLflow y que permite hacer inferencias.
	*	**Smarlint**: Interfaz web que permite hacer inferencias.


## Instrucciones
Clone el repositorio de git usando el siguiente comando en la consola de su sistema operativo:


```
$ git clone https://github.com/c4ttivo/MLOPS-Project3.git
```

Una vez ejecutado el comando anterior aparece el folder MLOPS-Project2. Luego es necesario ubicarse en el directorio de trabajo en el que se encuentra el archivo docker-compose.yml.


```
$ cd MLOPS-Project3/
```

Ahora es necesario construir los contenedores

```
$ echo -e "AIRFLOW_UID=$(id -u)" > .env
$ sudo docker-compose up airflow-init
$ sudo docker-compose up
```

En este paso se descarga las imágenes de acuerdo con lo especificado en el archivo docker-compose.yml.

<img src="https://github.com/c4ttivo/MLOPS-Project3/blob/main/img/console.png?raw=true" width="50%" height="50%" />

Una vez finalizada la creación de los contenedores, se debe poder ingresar a las aplicaciones de cada contenedor a través de las siguientes URLs:

http://10.43.101.155:8083/ </br>
<img src="https://github.com/c4ttivo/MLOPS-Project3/blob/main/img/minio.png?raw=true" width="50%" height="50%" /> </br>
http://10.43.101.155:8082/ </br>
<img src="https://github.com/c4ttivo/MLOPS-Project3/blob/main/img/mlflow.png?raw=true" width="50%" height="50%" /> </br>
http://10.43.101.155:8080/ </br>
<img src="https://github.com/c4ttivo/MLOPS-Project3/blob/main/img/airflow.png?raw=true" width="50%" height="50%" /> </br>

## Configuración

Los siguientes pasos permiten realizar la configuración del ambiente luego de ser desplegado.

1.	A continuación se debe configurar el bucket de S3, con el nombre **mlflows3** requerido por **MLflow**.

## Predicción

A través de la interfaz de FastAPI, es posible hacer predicciones usando el modelo almacenado y etiquetado @produccion.

http://10.43.101.155/docs </br>

![alt text](https://github.com/c4ttivo/MLOPS-Project2/blob/main/img/inference.png?raw=true)

