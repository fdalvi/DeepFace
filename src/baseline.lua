--  DeepFace

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
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Load pretrained model
net = torch.load(params.model)

-- Adjust network
net:remove() -- remove softmax
net:remove() -- remove fc8

heads = nn.ConcatTable()
for head_idx = 1,3 do
	-- Create local head
	local head = nn.Sequential()
	head:add(nn.Linear(4096, 2):float())
	head:add(nn.SoftMax():float())
	heads:add(head)
end

-- Add heads to the network
net:add(heads)
print(net)

-- Evaluate network on the input image
net:evaluate()
im = image.load(params.image,3,'float')
im = im*255
mean = {129.1863,104.7624,93.5940}
im_bgr = im:index(1,torch.LongTensor{3,2,1})
for i=1,3 do im_bgr[i]:add(-mean[i]) end
prob = net(im_bgr)

-- Print results
maxval,maxid_1 = prob[1]:max(1)
maxval,maxid_2 = prob[2]:max(1)
maxval,maxid_3 = prob[3]:max(1)
print(maxid_1)
print(maxid_2)
print(maxid_3)
