---
hide:
  - toc
---

# Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**unitorch** is a powerful package designed with a Foundation-Adapter architecture, providing a streamlined development process for unified models in various domains such as natural language understanding, natural language generation, computer vision, click-through rate prediction, multimodal learning, and reinforcement learning. The package is built on top of PyTorch, a popular deep learning framework, and seamlessly integrates with other well-known frameworks like transformers, peft, diffusers, and fastseq.

<hr/>

![Overview](overview.png)

<hr/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The architecture of unitorch consists of two main components: **Foundation Modules**, **Adapter Modules** and **Command Line Interface (CLI)**. 

**Foundation Modules**: focus on implementing the core functionality of the package and provide the basic functions required by the models. These modules serve as the building blocks for different workflows and are designed to be modular, efficient, and flexible.

**Adapter Modules**: act as adapters for the Foundation Modules, enabling them to support different workflows. Since different tasks or applications may have unique requirements, Adapter Modules provide the necessary interfaces and configurations to adapt the Foundation Modules to specific use cases. This modular approach allows for easy customization and extensibility of the package.

**Command Line Interface (CLI)**: defines the Running workflow to streamline the usage of unitorch. The CLI orchestrates the execution of the pipeline by calling the required Adapter Modules based on the pipeline design. This command line tool simplifies the process of training, evaluating, and deploying models, making it convenient for researchers and developers to experiment with different configurations and workflows.

* `unitorch-train` command is used to train models using the unitorch package. It enables you to specify the training data, model architecture, hyperparameters, and other configuration options. By Running this command, the package will utilize the specified data and parameters to train the model and optimize its performance based on the defined objective.
* `unitorch-infer` command is used for inference or prediction using trained models. Once a model has been trained using unitorch-train, you can employ this command to make predictions or generate outputs for new or unseen data. It takes the trained model and the input data as inputs and produces the predicted results using the learned patterns and knowledge captured during training.
* `unitorch-eval` command is used to evaluate the performance of trained models. It allows you to assess the quality and effectiveness of the model by comparing its predictions against the ground truth or reference data. This command typically computes various metrics, such as accuracy, precision, recall, F1 score, or other domain-specific metrics, to provide insights into the model's performance.
* `unitorch-script` command provides a convenient way to execute custom scripts or workflows using the unitorch package. It enables you to define and execute complex operations or experiments by writing scripts that leverage the functionalities of the unitorch library. This command allows for more flexibility and customization in using the package for specific research or development tasks.
* `unitorch-service` command is used to deploy a service for data serving, including hosting an HTTP server, serving models, or exposing APIs. This command allows you to define the endpoints and routes for your service, configure the behavior and responses, and integrate the necessary functionalities from the unitorch package.

<hr/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In addition to the CLI, unitorch also offers a simple import statement (import unitorch) that allows users to leverage the functionality of the package with just a single line of code. This import statement provides access to the state-of-the-art models and datasets supported by unitorch, without compromising performance or accuracy.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Overall, unitorch empowers developers and researchers to build unified models across different domains quickly and efficiently. By leveraging the Foundation-Adapter architecture and integrating seamlessly with popular frameworks, unitorch simplifies the development process and accelerates the deployment of advanced models in various applications.