# Nyquist explorer
 
## Setup
* Clone the repository.

`git clone https://github.com/karanrajagopalan/Nyquist-explorer.git`

You can choose one of the below methods to run the app

1. run the app from your local setup by installing the required version of python and the needed packages 
(or)
2. run the app as a docker container by building its docker image. You'll need to install docker in that case.

### Running locally
* Install python3.9.13 from  `https://www.python.org/downloads/release/python-3913/`
* Go to terminal / command prompt, install poetry

`pip install poetry`

`poetry install`

`poetry shell`

* Uncomment the last 2 lines of app/Nyquist_explorer.py

`    if __name__ == '__main__':
       app.run_server(debug=True, host ='0.0.0.0', port = 8050)`
* Run the app as 

    `python Nyquist_explorer.py`

### Running docker container
* If you want to run this as a docker container, then you need to build the docker image.
* Install docker. Follow the steps given in docker docs [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
* With the Nyquist_explorer as your current directory, Build the docker image with the command

 `docker build ./ -t nyqust_explorer`

* Run the container with the command below.
 `docker run -p 8050:8050 nyquist_exp`

#### NOTE: 
In the both the above cases, you'll be able to access the app from [http://localhost:8050](http://localhost:8050)

## Usage
Nyquist explorer is a simple vibration analysis tool with basic functionalities. You can use any nc (netCDF4) file as the input. You can visualize 
 1. Raw time domain plot
 2. Frequency spectral plot
 3. Apply FIR filter with some of the common window functions. Supported filter types
    a. Low pass
    b. High pass
    c. Band pass
 4. Visualize the results of the filter applied in time domain and the frequency plot.

Other features included,
 Visualizing the frequency spectrum for the specified timeline. USe the built in plotly zoom function in the time domain plot.
![image](https://user-images.githubusercontent.com/58655145/152064393-7f79bb10-c885-4c0b-8296-bdfb81df0ff4.png)
