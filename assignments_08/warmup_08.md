# Part 1: Warmup — Cloud Concepts

## Cloud Concepts Question 1
### What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core concept in cloud computing is that you rent resources from a provider -- storage, processing power, networking -- and pay for what you use, known as pay-as-you-go model. With cloud computing, the cloud provider owns and maintains the physical hardware, so you can rent the resources you need without having to manage the physical servers yourself.


## Cloud Concepts Question 2
### What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

•	Vertical scaling = making one machine bigger, increasing the size of CPU, GPU, and RAM
•	Horizontal scaling = adding more machines and splitting the work across them

### Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

### A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
Horizontal scaling fits this scenario because it needs to scale on demand (support many more users), so adding more servers to distribute the workload can really helpful.

### A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
Vertical scaling fits this scenario since it needs to increase the GPU and RAM.

### A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
Horizontal scaling fits this scenario since the pipeline needs to process many more files, and the work can be split across machines.


## Cloud Concepts Question 3
### Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.
•	Gmail: Saas, because it's a complete web app delievered directly to end-users without requiring them to manage the underlying infrastructure.
•	Azure Virtual Machines: Iaas, because users are responsible for setting up the operating system and software and applying security updates.
•	AWS S3 (Simple Storage Service): Iaas, because it provides scalable object storage.
•	GitHub Codespaces: Paas, because it provides a managed development environment for writing and running code.
•	Snowflake: Managed data platform, because it provides a complete set of services for storing, processing, analyzing, and sharing data.
•	Supabase: Baas, because it provides a managed PostgreSQL database along with backend services such as authentication, APIs, and storage.

### Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

Iaas = a virtual machine, some storage, a network connection -- and you set up everything yourself, just as you would on your own computer. You choose the operating system, install your software, configure your environment, and handle security updates. Example, AWS, Microsoft Azure.

Paas = You bring your application code, and the platform runs it for you. The provider manages the infrastructure, but you bring your own code. You deploy an application or a script, and the platform handles running it, scaling it, and keeping the underlying machine healthy. 
Examples: Azure App Service, AWS Elastic Beanstalk, Google App Engine.

Saas = The provider manages the infrastructure and the application itself, you simply use the finished software. You typically open a browser/app, log in, and use the service. You don't deploy or maintain the application's code. You don't think about servers at all.
Examples: Gmail, Google Docs, Dropbox, Salesforce.
 

## Cloud Concepts Question 4
### What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

Managed data platforms are integrated services where providers handle much of the underlying infrastructure, scaling, availability, and platform operations. When using a cloud provider like AWS or GCP directly, you need to set up the infrastructure and connect the resources needed to work with data.

You gain faster setup and less infrastructure management, but you give up some control and flexibility over the underlying infrastructure.


## Cloud Concepts Question 5
### The lesson names two situations where the cloud is probably not the right choice. What are they?

If your dataset fits comfortably on a single machine and you do not have massive compute demands, local processing is often faster and cheaper. This is often the best approach when setting up an initial prototype.
It may also not be the right choice when the learning curve and complexity of cloud infrastructure outweigh the benefits, especially for simple tasks or initial prototypes.


# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1
### Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

o	AWS: Broadest service ecosystem and strong scalability; commonly used by startups, enterprises, and large organizations.
o	GCP: Strong in data analytics, AI, and machine learning; commonly used by data-driven companies, tech organizations, and research teams.
o	Azure: Strong enterprise integration and Microsoft ecosystem support; commonly used by large enterprises, government organizations, and companies already using Microsoft technologies.


## Cloud Landscape Question 2
### The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

Access. Supabase is easier to access because students can create their own accounts in a few minutes without needing organizational provisioning or waiting for invitations.

Pedagogical fit. Azure uses a service named Azure Blob Storage to store files by path, which does not work with tables and rows like a relational database. The pipeline in this course build in weeks 9–11 maps naturally onto tables and queries.

Pipeline coherence. Supabase makes it easy to represent the raw and enriched stages of the ETL pipeline as related tables, you can query either table at any point to verify what's in it, which fits the ETL pipeline built in this course.

### Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?
When working on a real-world project, I don't have to commit to one cloud provider for everything. I should evaluate each tool based on the project's requirements, considering factors such as cost, scalability, control, ease of use, integration, and vendor lock-in.


## Cloud Landscape Question 3
### For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

1. You need to store 10 TB of image files and retrieve them by filename from any machine.
Object storage: AWS S3 from AWS provider.
2. You need to run an ML training job on a GPU for four hours, then shut it down.
GPU compute: Amazon EC2 from Amazon Web Services (AWS)
3. You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
Serverless compute: AWS Lambda from AWS provider.
4. You need to send structured data to a large language model and get a text response back.
LLM API: OpenAI API.



## Cloud Landscape Question 4
### The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

I would build a Python data analysis project using a dataset of different cereals (from python-100 project), including their calories, sugar, protein, and rating. The project would analyze which nutritional factors are related to higher cereal ratings and produce visualizations of the results.

Stack:
Object storage → AWS → S3: Store the cereal dataset.
Compute → GCP → Compute Engine: Run the Python data analysis.

Using one provider can simplify management, integration, and security, but you may be limited in choosing the best service from different providers based on factors like price.


