# 我当前正在开发一个集成打标与训练的lora训练器
## 页面1：数据集管理页-导航显示
数据集管理页是一个列表页，使用列表的形式展示我创建的数据集，当我点击创建新数据集按钮，直接在列表中增加一条数据集，我可以点击列表容器上的设置图标来
编辑数据集的信息包括数据集名称数据集类型，类型包括【图像】【视频】【图像+控制图像】，在列表 上要显示创建时间，数据集类型数据集数量这些信息
当我点击查看详情进入下一级页面
## 页面2：数据集详情页
### 数据集详情页分为两个页面分别是【数据集管理】和【标签编辑】，用户可以选择对应的卡片进行批量操作，可以全选也可以取消全选
数据集打标页包含自动打标按钮，和一个用于展示数据集的容器，数据集以卡片的形式展示，每个卡片都包含数据集图片，图片文件名，标签内容
用户可以在内容部分实时的编辑并保存，卡片比例固定，宽度按照屏幕或者app宽度动态适应，并且自适应调整卡片列数
标签编辑页面
标签编辑页面按照列表来展示，展示每张图片对应的标签，标签按照打标文案的逗号来进行分割，用户可以拖拽来对标签进行排序，顶部的功能包括新增标签，删除标签
批量编辑例如批量删除或者批量替换等
## 页面3：创建训练任务-导航显示
进入创建任务后直接展示创建训练任务的界面，用户可以选择任务类型，例如【qwen-image-lora】【kontext-lora】【wan2.2-lora】等任务
训练器使用的是[https://github.com/kohya-ss/musubi-tuner]  连接是git仓库地址，
这里我给出训练的步骤以及相关的命令：
Qwen-Image
Overview
This document describes the usage of the Qwen-Image architecture within the Musubi Tuner framework. Qwen-Image is a text-to-image generation model.
This feature is experimental.

Download the model
You need to download the DiT, VAE, and Text Encoder (Qwen2.5-VL) models.

DiT, Text Encoder (Qwen2.5-VL): For DiT and Text Encoder, download split_files/diffusion_models/qwen_image_bf16.safetensors and split_files/text_encoders/qwen_2.5_vl_7b.safetensors from https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI, respectively. The fp8_scaled version cannot be used.

VAE: For VAE, download vae/diffusion_pytorch_model.safetensors from https://huggingface.co/Qwen/Qwen-Image. ComfyUI's VAE weights cannot be used.

Pre-caching
Latent Pre-caching
Latent pre-caching uses a dedicated script for Qwen-Image.

python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
Uses qwen_image_cache_latents.py.
The --vae argument is required.

Text Encoder Output Pre-caching
Text encoder output pre-caching also uses a dedicated script.

python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16
Uses qwen_image_cache_text_encoder_outputs.py.
Requires the --text_encoder (Qwen2.5-VL) argument.
Use the --fp8_vl option to run the Text Encoder in fp8 mode for VRAM savings for <16GB GPUs.
Training
Training uses a dedicated script qwen_image_train_network.py.

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift \
    --weighting_scheme none --discrete_flow_shift 3.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
Uses qwen_image_train_network.py.
Requires specifying --dit, --vae, and --text_encoder.
The LoRA network for Qwen-Image (networks.lora_qwen_image) is automatically selected.
--mixed_precision bf16 is recommended for Qwen-Image training.
Memory saving options like --fp8_base and --fp8_scaled (for DiT), and --fp8_vl (for Text Encoder) are available.
--gradient_checkpointing is available for memory savings.
--fp8_vl is recommended for GPUs with less than 16GB of VRAM.

--sdpa uses PyTorch's scaled dot product attention. Other options like --xformers and --flash_attn are available. flash3 cannot be used currently.

If you specify --split_attn, the attention computation will be split, slightly reducing memory usage. Please specify --split_attn if you are using anything other than --sdpa.

--timestep_sampling allows you to choose the sampling method for the timesteps. shift with --discrete_flow_shift is the default. qwen_shift is also available. qwen_shift is a same method during inference. It uses the dynamic shift value based on the resolution of each image (typically around 2.2 for 1328x1328 images).

--discrete_flow_shift is set quite low for Qwen-Image during inference (as described), so a lower value than other models may be preferable.

The appropriate settings for each parameter are unknown. Feedback is welcome.

VRAM Usage Estimates with Memory Saving Options
For 1024x1024 training with the batch size of 1, --mixed_precision bf16 and --gradient_checkpointing is enabled and --xformers is used.

options	VRAM Usage
no	42GB
--fp8_base --fp8_scaled	30GB
+ --blocks_to_swap 16	24GB
+ --blocks_to_swap 45	12GB
64GB main RAM system is recommended with --blocks_to_swap.

If --blocks_to_swap is more than 45, the main RAM usage will increase significantly.
以上就是使用musubi-tuner进行训练的步骤，不同的模型可能加载的参数不太一样，
此外页面的交互处理你需要根据需要输入的信息来进行分组展示，做成下拉框和文本输入
#### 页面展示包括
页面顶部显卡类型，显卡显存大小以及占用进度
0 任务类型设置，任务标题，显存预设选择例如高中低显存，分辨对应不同的参数以及缓存块等等，当用户编辑后自动变为自定义，并根据预估实时变更顶部的显存占用预警
1 模型文件设置： 底模，clip, vae的路径填写 ，这个模型路径我也希望在设置里进行添加
2 数据集设置： 使用哪个数据集，需要能选择我数据集管理创建的数据集， 设置这个数据集重复次数，训练分辨率，是否开启分桶，开启的话则展示分桶相关设置例如最大最小，以及分桶间隔的分辨率 
3 学习率优化器与调度器 等等参数
4过程中采样： 采样prompt，每多少步进行采样or多少轮采样，采样的一些其他设置例如分辨率等等 
5使用哪个gpu 
6高级设置： 主要是一些其他内容具体要看训练什么模型来定 
根据刚刚的进行一个分组展示，当设置完毕点击创建任务则自动跳转到训练任务管理tab
## 页面3：训练任务管理-导航显示
训练任务管理也是一个列表，列表上显示当前任务标题，创建时间，状态，训练进度，剩余时间，等一些参数，可以点击查看详情，也可以删除，但是训练中点击删除会
弹窗进行询问，删除任务需要先停止任务，是否停止任务并删除
## 页面4：训练任务详情
这个可以参考aitoolkit的设计，分多个容器展示左侧容器第一个row展示训练任务名称，使用的gpu，训练的速度以及剩余时间，
第二个row显示终端窗口，展示训练中终端的输出，右侧的容器展示当前机器gpu的信息，温度风扇转速，gpu占用，显存占用，显存频率，功率等，再有一个容器展示输出内容包括当前训练过程采样的结果和已经保存的checkpoints文件 
任务过程中采样的结果实时与保存的模型的显示在页面中，并且有一个按钮，点击可以打开对应的文件


## 页面5：设置-导航显示
主要包括不同模型的文件路径设置，后续打标以及标签处理的方式，例如调用本地模型还是api，api的话需要填写token等


用户的操作路径为，上传数据集，自动打标，优化调整数据集，创建训练任务，开始训练，查看训练结果等
以上就是当前软件的规划