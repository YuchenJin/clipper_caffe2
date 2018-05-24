import requests, json, numpy as np
DEFAULT_LABEL = []

def register_application(name, input_type, default_output,
                             slo_micros, ip="localhost", port="1338"):
    url = "http://{ip}:{port}/admin/add_app".format(ip=ip, port=port)
    req_json = json.dumps({
                            "name": name,
                            "input_type": input_type,
                            "default_output": default_output,
                            "latency_slo_micros": slo_micros
                          })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    if r.status_code != requests.codes.ok:
        msg = "Received error status code: {code} and message: {msg}".format(
        code=r.status_code, msg=r.text)
        print msg
    else:
        print("Application {app} was successfully registered".format(app=name))


def register_model(name,
                   version,
                   input_type,
                   image=None,
                   labels=None,
                   batch_size=-1,
                   ip="localhost",
                   port="1338"):
    version = str(version)
    url = "http://{ip}:{port}/admin/add_model".format(ip=ip, port=port)
    if labels is None:
        labels = DEFAULT_LABEL
    req_json = json.dumps({
        "model_name": name,
        "model_version": version,
        "labels": labels,
        "input_type": input_type,
        "container_name": image,
        "model_data_path": "DEPRECATED",
        "batch_size": batch_size
    })

    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    if r.status_code != requests.codes.ok:
        msg = "Received error status code: {code} and message: {msg}".format(
            code=r.status_code, msg=r.text)
        print msg
    else:
        print(
            "Successfully registered model {name}:{version}".format(
                name=name, version=version))


def link_model_to_app(app_name, model_name, ip="localhost", port="1338"):
    url = "http://{ip}:{port}/admin/add_model_links".format(ip=ip, port=port)
    req_json = json.dumps({
        "app_name": app_name,
        "model_names": [model_name]
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    if r.status_code != requests.codes.ok:
        msg = "Received error status code: {code} and message: {msg}".format(
            code=r.status_code, msg=r.text)
        print msg
    else:
        print(
            "Model {model} is now linked to application {app}".format(
                model=model_name, app=app_name))


#register_application(name="hello-world", input_type="strings", default_output="-1.0", slo_micros=100000)
#register_model("clipper-noop", 1, "strings", "clipper:caffe2")
#link_model_to_app("hello-world", "clipper-noop")
