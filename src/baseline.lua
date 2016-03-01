--  DeepFace

require 'torch'
require 'image'
require 'nn'

-- Setup command line parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a baseline network')
cmd:text()
cmd:text('Options')
cmd:option('-model','./vgg_face_torch/VGG_FACE.t7','path to pretrained model')
cmd:option('-image','./vgg_face_torch/ak.png' ,'Image to run forward pass on')
cmd:option('-num_iters', 1000,'Number of iterations to train')
cmd:option('-learning_rate', 1e-1, 'Learning rate on training')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Load pretrained model
net = copy(torch.load(params.model))
-- net = require(params.model)
-- Adjust network
net:remove() -- remove softmax
net:remove() -- remove fc8

heads = nn.ConcatTable()
for head_idx = 1,3 do
	-- Create local head
	local head = nn.Sequential()
	head:add(nn.Linear(4096, 2):float())
	head:add(nn.LogSoftMax():float())
	heads:add(head)
end

-- Add heads to the network
net:add(heads)
-- print(net)

criterion = nn.ClassNLLCriterion() --loss function

-- preprocessing input image
im = image.load(params.image,3,'float')
im = im*255
mean = {129.1863,104.7624,93.5940}
im_bgr = im:index(1,torch.LongTensor{3,2,1})
for i=1,3 do 
	im_bgr[i]:add(-mean[i]) 
end

dataset = torch.Tensor(1,3,224,224)
dataset:select(1,1):copy(im_bgr:double())
labels = torch.Tensor(1,1)
labels:fill(1)

--define function to get next batch of training images
function nextBatch()
	local inputs, targets = {}, {}

	--get a batch of inputs
	table.insert(inputs, dataset:index(1,1))	
	-- fill the batch of targets
	table.insert(targets, labels:index(1,1))
	return inputs, targets
end

-- get weights and loss wrt weights from the model
net:zeroGradParameters()
print(net:parameters())
w, dl_dw = net:getParameters()
print(w)
print(dl_dw)
-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. weigths is the vector of trainable weights,
-- it extracts a mini_batch via the nextBatch method
feval = function(w_new)
	-- copy the weight if are changed
	if w ~= w_new then
		w:copy(w_new)
	end
	-- select a training batch
	local inputs, targets = nextBatch()
	-- reset gradients (gradients are always accumulated, to accommodate
	-- batch methods)
	dl_dw:zero()

	-- evaluate the loss function and its derivative with respect to w, given a mini batch
	local prediction = net:forward(inputs)
	local loss_w = criterion:forward(prediction, targets)
	net:backward(inputs, criterion:backward(prediction, targets))

	return loss_w, dl_dw
end

--training 
optim_state = {
	learningRate = params.learning_rate 
}
for t = 1, params.num_iters do
	_, losses = optim.adam(feval, w, optim_state)
end


-- Evaluate network on the input image
net:evaluate()
prob = net(im_bgr)

-- Print results
maxval, maxid_1 = prob[1]:max(1)
maxval, maxid_2 = prob[2]:max(1)
maxval, maxid_3 = prob[3]:max(1)
print(maxid_1)
print(maxid_2)
print(maxid_3)
