
build:
	docker build -t vexas_classification .

launch:
	docker run -v ${PWD}/:/code -ti vexas_classification /bin/bash

launch-gpu:
	docker run --runtime nvidia -v ${PWD}/:/code -ti vexas_classification /bin/bash

launch-jupyter:
	docker run -v ${PWD}/:/code --runtime nvidia \
			   -ti -p 9999:9999 vexas_classification jupyter notebook --ip 0.0.0.0 --port 9999 --allow-root --no-browser
