# Part 3: Project

## Part A : Supabase Setup
My Supabase project is set up successfully, I created the weather_raw and weather_enriched tables. Both are listed in the Table Editor, and I also disabled Row Level Security as instructed.

## Part B: Cloud Cost Analysis
Scenario A: The t3.micro EC2 instance costs approximately $1.66 per month for 160 hours of use. I was surprised that the cost was so low for a lightweight workload.

Scenario B: The total cost is approximately $2,382.18 per month. This includes $2,233.80 for the p3.2xlarge EC2 instance, $124.83 for the RDS db.m5.large instance, and $23.55 for 1 TB of S3 Standard storage. I was surprised by how much the GPU instance contributes to the total cost.

Something interesting I found: Exploring the AWS Pricing Calculator showed me that changing the instance type, usage hours, or pricing option can make a large difference in the monthly cost. I also noticed that AWS can recommend lower-cost instance types, so it is important to check that the selected instance matches the requirements.

Comparison: Scenario B is dramatically more expensive than Scenario A, mainly because of the GPU instance. This shows that a GPU instance is worth using when a workload needs significant GPU processing power, such as heavy analytics or machine learning, but for lightweight workloads, a smaller CPU-based instance is much more cost-effective.

## Video Link
### Note: If the video does not load in Microsoft Edge, please open the link in Google Chrome. The recording has been tested and plays successfully in Chrome.

https://drive.google.com/file/d/13zMvaF27pr3o6iVO2L_F92YuMnaeo5gx/view?usp=sharing




