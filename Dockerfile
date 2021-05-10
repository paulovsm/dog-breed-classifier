FROM opencvcourses/opencv-docker

ADD app/ app/

RUN cd app/ && pip install -r requirements.txt --no-cache-dir

WORKDIR /home/app

CMD ["python", "web.py" ]