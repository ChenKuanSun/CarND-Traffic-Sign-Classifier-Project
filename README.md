## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used the deep neural network and convolutional neural networks I learned in the course to classify traffic signs. I trained and validated the model to classify traffic sign images using the [German Traffic Sign Dataset] (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model training, I found some models on the Internet to find some models of the German traffic signs.

[//]: # (Image References)

[image1]: ./output/The_count_of_each_sign.png "The count of each sign"
[image2]: ./output/Random_Training_Data.png "Random_Training_Data"
[image3]: ./output/Random_Test_Data.png "Random_Test_Data"
[image4]: ./output/Random_Generated_Data.png "Random_Generated_Data"
[image5]: ./output/Random_preprocess_Data.png "Random_preprocess_Data"
[image6]: ./output/Validation_Accuracy.png "Validation_Accuracy"


[image7]: ./output/img1.png "img1"
[image8]: ./output/img2.png "img2"
[image9]: ./output/img3.png "img3"
[image10]: ./output/img4.png "img4"
[image11]: ./output/img5.png "img5"


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


---
#### Load the data set

First of all, I first imported the pickle module to load the data of this project.

Data Path: ./traffic-signs-data/

```python
import pickle

training_file = "./traffic-signs-data/train.p"
validation_file= "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

#### Explore, summarize and visualize the data set

Let's first understand the overall size of the data.
```python
#print(X_train.shape)
#print(len(labels_title))
label_counts = collections.Counter(y_train)
labels_title = [x[0] for x in sorted(label_counts.most_common())]

n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
n_classes = len(labels_title)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
Result：
Number of training examples = 258000
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

Next I use BAR to visualize my data and randomly display training and test data to check if the distribution is consistent.

```python
with open('signnames.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    label_define = []
    next(rows)
    for row in rows:
        label_define.append(row[1])

labels_total = [x[1] for x in sorted(label_counts.most_common())]

#print(labels_title)
#print(labels_total)

plt.bar(labels_title,labels_total, 1, color='br',alpha=0.5)
plt.title('The count of each sign') 
plt.xlabel('Labels')
plt.ylabel('Numbers of Labels')
plt.tight_layout()
plt.savefig("./output/The_count_of_each_sign.png")
plt.show()

print("**********Random Training Data!!**************")

#Show example of dataset
H_imgs = 4
V_imgs = 3
fig, axs = plt.subplots(H_imgs,V_imgs, figsize=(20, 25))
axs = axs.ravel()
for i in range(0 , H_imgs*V_imgs):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    axs[i].imshow(image, cmap='gray', aspect='auto')
    axs[i].set_title(label_define[y_train[index]], fontsize=20)
plt.savefig("./output/Random_Training_Data.png")
plt.show()

print("**********Random Test Data!!**************")

H_imgs = 4
V_imgs = 3
fig, axs = plt.subplots(H_imgs,V_imgs, figsize=(20, 25))
axs = axs.ravel()
for i in range(0 , H_imgs*V_imgs):
    index = random.randint(0, len(X_test))
    image = X_test[index].squeeze()
    axs[i].imshow(image, cmap='gray', aspect='auto')
    axs[i].set_title(label_define[y_test[index]], fontsize=20)
plt.savefig("./output/Random_Test_Data.png")
plt.show()
```
Result：

![alt text][image1]
![alt text][image2]
![alt text][image3]

Because the training data is too small, it is not conducive to training the model, so I tried to create training data by means of perspective transformation and changing the RGB space.

The function of perspective transformation is very useful information found on the Internet. Here is the link https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9


```python
every_class_target = 6000
IMAGE_SIZE = 32
def transform_image(img,ang_range,shear_range,trans_range):
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # She
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    return img

def gray_img(img):
    mean = np.mean(img, axis=2, keepdims=True).astype(np.uint8)
    mean_data = np.concatenate((mean,mean,mean),axis=2)
    return np.array(mean_data, np.int32)

def generate_images():
    new_x_train = []
    new_y_train = []
    index = 0
    for label_total in labels_total:
        images = []
        #gather images in a list
        for i in range(0, len(X_train)):
            if index == y_train[i]:
                images.append(X_train[i])

        while label_total < every_class_target:
            image = random.choice(images)
            pres = [transform_image(image,20,10,5), transform_image(image,20,10,5), transform_image(image,20,10,5), gray_img(image)]
            #chose 3:1
            image = random.choice(pres)
            #image = transform_image(image,20,10,5)
            if new_x_train:
                new_x_train.append(image)
            else:
                new_x_train = [image]
            new_y_train.append(index)
            label_total += 1
        index += 1
    return new_x_train, new_y_train

print('Generating images')
new_x_train, new_y_train = generate_images()

print("**********Random Generated Data!!**************")

H_imgs = 4
V_imgs = 3
fig, axs = plt.subplots(H_imgs,V_imgs, figsize=(20, 25))
axs = axs.ravel()
for i in range(0 , H_imgs*V_imgs):
    index = random.randint(0, len(new_x_train))
    image = np.array(new_x_train[index].squeeze(),np.uint8)
    axs[i].imshow(image, cmap='gray', aspect='auto')
    axs[i].set_title(label_define[new_y_train[index]], fontsize=20)
plt.show()

print('input images: ' + str(len(X_train)))

print('generated images: ' + str(len(new_x_train)))

X_train = np.append(X_train, new_x_train).reshape((-1,32,32,3))

y_train = np.append(y_train, new_y_train)

print('Total training images: ' + str(len(X_train)))

```

Then show the generated results

![alt text][image4]

Normalized data:
```python
def preprocess(data):
    mean = np.mean(data)
    std = np.std(data)
    imgs = center_normaize(data, mean, std)
    return data
def center_normaize(data, mean, std):
    data = data.astype('float32')
    data -= mean
    data /= std
    return data
X_train = preprocess(X_train)
X_test = preprocess(X_test)
X_valid = preprocess(X_valid)

X_train, y_train = shuffle(X_train, y_train)
print("**********Random Training Data!!**************")

#Show example of dataset
H_imgs = 4
V_imgs = 3
fig, axs = plt.subplots(H_imgs,V_imgs, figsize=(20, 25))
axs = axs.ravel()
for i in range(0 , H_imgs*V_imgs):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    axs[i].imshow(image, cmap='gray', aspect='auto')
    axs[i].set_title(label_define[y_train[index]], fontsize=20)
plt.show()
```
Then show the preprocess data results:

![alt text][image5]

#### Design, train and test a model architecture

Because I designed the model for the first time, I tried to modify it into my model with LeNet.

Then I used it three times at a time, because I have Nvidia GTX 1070 so I added one more layer.

<table>
	<thead>
		<tr>
			<th style="text-align:center">Layer</th>
			<th style="text-align:center">Description</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td style="text-align:center">Input</td>
			<td style="text-align:center">32x32x3 RGB image</td>
		</tr>
		<tr>
			<td style="text-align:center">Convolution 3x3</td>
			<td style="text-align:center">1x1 stride, same padding, outputs 32x32x16</td>
		</tr>
		<tr>
			<td style="text-align:center">LEAKY_RELU</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Max pooling 2x2</td>
			<td style="text-align:center">2x2 stride,  outputs 16x16x16</td>
		</tr>
		<tr>
			<td style="text-align:center">Convolution 3x3</td>
			<td style="text-align:center">1x1 stride, same padding, outputs 16x16x32</td>
		</tr>
		<tr>
			<td style="text-align:center">LEAKY_RELU</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Max pooling 2x2</td>
			<td style="text-align:center">2x2 stride, outputs 8x8x32</td>
		</tr>
		<tr>
			<td style="text-align:center">Convolution 3x3</td>
			<td style="text-align:center">1x1 stride, same padding, outputs 8x8x64</td>
		</tr>
		<tr>
			<td style="text-align:center">LEAKY_RELU</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Max pooling 3x3</td>
			<td style="text-align:center">2x2 stride, outputs 3x3x64</td>
		</tr>
		<tr>
			<td style="text-align:center">Flatten</td>
			<td style="text-align:center">output 576</td>
		</tr>
		<tr>
			<td style="text-align:center">Fully connected</td>
			<td style="text-align:center">output 120</td>
		</tr>
		<tr>
			<td style="text-align:center">LEAKY_RELU</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Dropout</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Fully connected</td>
			<td style="text-align:center">outout 84</td>
		</tr>
		<tr>
			<td style="text-align:center">LEAKY_RELU</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Dropout</td>
			<td style="text-align:center"></td>
		</tr>
		<tr>
			<td style="text-align:center">Fully connected</td>
			<td style="text-align:center">output 43</td>
		</tr>
		<tr>
			<td style="text-align:center">Softmax</td>
			<td style="text-align:center"></td>
		</tr>
	</tbody>
</table>



```python
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    Conv1_filter = 3
    Conv1_feature = 16
    Conv2_filter = 3
    Conv2_feature = 32
    Conv3_filter = 3
    Conv3_feature = 64
    Fc1_feature = 120
    Fc2_feature = 84
    #Layer 1: Convolutional.
    print("Input Data shape:"+ str(x.shape))
    conv1_W = tf.Variable(tf.truncated_normal(shape=(Conv1_filter, Conv1_filter, int(x.shape[3]), Conv1_feature), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(Conv1_feature))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True) + conv1_b

    #Activation.
    conv1 = tf.nn.relu(conv1)

    #Pooling. Input =24x24x48.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    print("Conv1.shape :" + str(conv1.shape))
    
    #Layer 2: Convolutional.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(Conv2_filter, Conv2_filter, Conv1_feature, Conv2_feature), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(Conv2_feature))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True) + conv2_b
    
    #Activation.
    conv2 = tf.nn.relu(conv2)

    #Pooling. Input = 8x8x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    print("Conv2.shape :" + str(conv2.shape))

    
    #Layer 2: Convolutional. input 4x4x128 
    conv3_W = tf.Variable(tf.truncated_normal(shape=(Conv3_filter, Conv3_filter, Conv2_feature, Conv3_feature), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(Conv3_feature))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True) + conv3_b
    
    #Activation.
    conv3 = tf.nn.relu(conv3)

    #Pooling. 
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    print("Conv3.shape :" + str(conv3.shape))
    
    
    #Flatten. 
    fc0   = flatten(conv3)
    #fc0   = flatten(conv2)
    
    print("Fc0.shape :" + str(fc0.shape))
    
    #Layer 3: Fully Connected. 
    fc1_W = tf.Variable(tf.truncated_normal(shape=(int(fc0.shape[1]), Fc1_feature), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(Fc1_feature))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    #Activation.
    fc1    = tf.nn.relu(fc1)
    
    print("Fc1.shape :" + str(fc1.shape))
    
    #Layer 4: Fully Connected. 
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(Fc1_feature, Fc2_feature), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(Fc2_feature))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    #Activation.
    fc2    = tf.nn.relu(fc2)
    
    print("Fc2.shape :" + str(fc2.shape))
    
    #Layer 5: Fully Connected. 
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(Fc2_feature, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

```
After designing the model, it starts to be set.

EPOCHS = 100

BATCH_SIZE = 1024

Learning rate = 0.0005

cross_entropy = softmax

optimizer = AdamOptimizer

Because there are more than one category, use softmax instead of logits.

Use ADAM to avoid gradient descent stopping at the peak

Calculate the accuracy of each Epochs and plot it so that it can be very useful when troubleshooting.


```python
x = tf.placeholder(tf.float32, (None, int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3])))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

p = IntProgress()

EPOCHS = 100
BATCH_SIZE = 1024
rate = 0.001

p.max = EPOCHS
p.description = 'Progress:'
p.bar_style = 'info'

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    
    validation_accuracy_group = []
    
    display(p)
    
    for i in range(EPOCHS):
        p.value = i+1
        X_train, y_train = shuffle(X_train, y_train)
        
        for offset in range(0, num_examples, BATCH_SIZE):
            
            end = offset + BATCH_SIZE
            
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        
        validation_accuracy = evaluate(X_valid, y_valid)
        
        validation_accuracy_group.append(validation_accuracy*100)
    for i in range(1,EPOCHS+1):
        print("EPOCH"+str(i)+" Validation Accuracy=  {:.1f}".format(validation_accuracy_group[i-1]))
    plt.plot(np.array(range(1, EPOCHS+1)), np.array(validation_accuracy_group))
    plt.xlabel('Validation Accuracy')
    plt.ylabel('EPOCHS')
    plt.savefig("./output/Validation_Accuracy.png")
    plt.show()
    saver.save(sess, './mynet')
    print("Model saved")
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
 ```
Result：

Input Data shape:(?, 32, 32, 3)
Conv1.shape :(?, 16, 16, 32)
Conv2.shape :(?, 8, 8, 128)
Conv3.shape :(?, 4, 4, 256)
Fc0.shape :(?, 4096)
Fc1.shape :(?, 120)
Fc2.shape :(?, 84)
Training...

EPOCH1 Validation Accuracy=  4.8

EPOCH2 Validation Accuracy=  3.4

EPOCH3 Validation Accuracy=  1.4

EPOCH4 Validation Accuracy=  3.4

EPOCH5 Validation Accuracy=  5.4

EPOCH6 Validation Accuracy=  0.7

EPOCH7 Validation Accuracy=  1.4

EPOCH8 Validation Accuracy=  1.4

EPOCH9 Validation Accuracy=  1.3

EPOCH10 Validation Accuracy=  1.3

EPOCH11 Validation Accuracy=  2.0

EPOCH12 Validation Accuracy=  2.0

EPOCH13 Validation Accuracy=  0.7

EPOCH14 Validation Accuracy=  2.7

EPOCH15 Validation Accuracy=  2.7

EPOCH16 Validation Accuracy=  0.7

EPOCH17 Validation Accuracy=  1.9

EPOCH18 Validation Accuracy=  7.4

EPOCH19 Validation Accuracy=  18.6

EPOCH20 Validation Accuracy=  45.8

EPOCH21 Validation Accuracy=  65.6

EPOCH22 Validation Accuracy=  74.9

EPOCH23 Validation Accuracy=  82.4

EPOCH24 Validation Accuracy=  86.6

EPOCH25 Validation Accuracy=  86.5

EPOCH26 Validation Accuracy=  89.0

EPOCH27 Validation Accuracy=  89.2

EPOCH28 Validation Accuracy=  90.4

EPOCH29 Validation Accuracy=  90.2

EPOCH30 Validation Accuracy=  91.8

EPOCH31 Validation Accuracy=  90.5

EPOCH32 Validation Accuracy=  90.5

EPOCH33 Validation Accuracy=  92.2

EPOCH34 Validation Accuracy=  92.3

EPOCH35 Validation Accuracy=  92.1

EPOCH36 Validation Accuracy=  92.9

EPOCH37 Validation Accuracy=  93.0

EPOCH38 Validation Accuracy=  93.4

EPOCH39 Validation Accuracy=  92.9

EPOCH40 Validation Accuracy=  93.2

EPOCH41 Validation Accuracy=  92.9

EPOCH42 Validation Accuracy=  93.1

EPOCH43 Validation Accuracy=  94.0

EPOCH44 Validation Accuracy=  95.0

EPOCH45 Validation Accuracy=  94.6

EPOCH46 Validation Accuracy=  94.2

EPOCH47 Validation Accuracy=  95.0

EPOCH48 Validation Accuracy=  95.3

EPOCH49 Validation Accuracy=  95.6

EPOCH50 Validation Accuracy=  95.2

EPOCH51 Validation Accuracy=  96.3

EPOCH52 Validation Accuracy=  95.3

EPOCH53 Validation Accuracy=  96.0

EPOCH54 Validation Accuracy=  96.2

EPOCH55 Validation Accuracy=  95.6

EPOCH56 Validation Accuracy=  95.9

EPOCH57 Validation Accuracy=  95.1

EPOCH58 Validation Accuracy=  96.3

EPOCH59 Validation Accuracy=  95.0

EPOCH60 Validation Accuracy=  95.4

EPOCH61 Validation Accuracy=  96.1

EPOCH62 Validation Accuracy=  95.8

EPOCH63 Validation Accuracy=  96.5

EPOCH64 Validation Accuracy=  96.2

EPOCH65 Validation Accuracy=  95.4

EPOCH66 Validation Accuracy=  96.3

EPOCH67 Validation Accuracy=  97.1

EPOCH68 Validation Accuracy=  96.9

EPOCH69 Validation Accuracy=  95.9

EPOCH70 Validation Accuracy=  96.1

EPOCH71 Validation Accuracy=  97.0

EPOCH72 Validation Accuracy=  96.2

EPOCH73 Validation Accuracy=  96.6

EPOCH74 Validation Accuracy=  97.1

EPOCH75 Validation Accuracy=  97.0

EPOCH76 Validation Accuracy=  97.4

EPOCH77 Validation Accuracy=  97.4

EPOCH78 Validation Accuracy=  97.5

EPOCH79 Validation Accuracy=  97.6

EPOCH80 Validation Accuracy=  96.5

EPOCH81 Validation Accuracy=  96.9

EPOCH82 Validation Accuracy=  97.0

EPOCH83 Validation Accuracy=  97.0

EPOCH84 Validation Accuracy=  97.0

EPOCH85 Validation Accuracy=  96.9

EPOCH86 Validation Accuracy=  97.1

EPOCH87 Validation Accuracy=  97.1

EPOCH88 Validation Accuracy=  97.3

EPOCH89 Validation Accuracy=  96.2

EPOCH90 Validation Accuracy=  96.8

EPOCH91 Validation Accuracy=  97.4

EPOCH92 Validation Accuracy=  97.2

EPOCH93 Validation Accuracy=  97.2

EPOCH94 Validation Accuracy=  97.6

EPOCH95 Validation Accuracy=  97.1

EPOCH96 Validation Accuracy=  97.5

EPOCH97 Validation Accuracy=  97.7

EPOCH98 Validation Accuracy=  96.8

EPOCH99 Validation Accuracy=  96.8

EPOCH100 Validation Accuracy=  97.9

![alt text][image6]



Model saved

INFO:tensorflow:Restoring parameters from .\mynet

Test Accuracy = 0.964


#### Use the model to make predictions on new images

Because I am not a German, I used Google Images to search a few traffic signs to test the model.
Of course, the picture is also subject to pre-processing to predict.

```python
for image_name in os.listdir("testimage/"):
    
    img = mpimg.imread("testimage/"+image_name)
    
    resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    
    X_data = np.array(resized_img, dtype = np.float32).reshape((-1,32,32,3))

    X_data = preprocess(X_data)
    
    fig, axs = plt.subplots(1,2, figsize=(20, 10))
    axs = axs.ravel()
    index = random.randint(0, len(new_x_train))
    fig.suptitle("img" + str(index), fontsize=20)
    
    axs[0].imshow(img, cmap='gray', aspect='auto')
    axs[0].set_title("Input_img", fontsize=15)
    
    axs[1].imshow(np.array(X_data.squeeze(),np.uint8), cmap='gray', aspect='auto')
    axs[1].set_title("Preprocess_img", fontsize=15)
    
    plt.savefig("./output/img" + str(index) + ".png")
    plt.show()
    
    index += 1
    evaluate_external_image(X_data)
    
    print("==========================================:D=========================================================")
```
Because the project requires "Analyze the softmax probabilities of the new images", I also added the chance when I predicted.
```python
def evaluate_external_image(X_data):
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        
        prediction=tf.nn.softmax(logits)
        
        sign_chance = sess.run([prediction],feed_dict={x:X_data})
        
        sign_chance = sign_chance[0][0]
        for i in range(0, 43):
            if sign_chance[i] > 0.01:
                print( str(int(sign_chance[i]*100)) + '%: ' + label_define[i] )
```

Result:

![alt text][image7]
INFO:tensorflow:Restoring parameters from .\mynet

100%: Right-of-way at the next intersection

==========================================:D=========================================================
![alt text][image8]
INFO:tensorflow:Restoring parameters from .\mynet

100%: Speed limit (60km/h)

==========================================:D=========================================================
![alt text][image9]
INFO:tensorflow:Restoring parameters from .\mynet

100%: Keep right

==========================================:D=========================================================
![alt text][image10]
INFO:tensorflow:Restoring parameters from .\mynet

100%: Roundabout mandatory

==========================================:D=========================================================
![alt text][image11]
INFO:tensorflow:Restoring parameters from .\mynet

100%: Stop

==========================================:D=========================================================


#### Summarize the results with a written report
When I first started the project, I didn't know where to start. Later, I followed the instructions from the LeNet network and took another time to review Andrew Ng's course at coursera.
Link: https://www.coursera.org/account/accomplishments/specialization/WZPSYNTR8PSH
In addition, I went to GOOGLE to find a lot of CNN materials, and later I really understood the whole architecture.
Then after a long training and finding information, I finally found the best answer.
Although there are still some problems in forecasting, I believe that if the amount of data is a little more and there are good quality photos, you can train a better model.
I can also add deeper networks to train and try to adjust DropOut to avoid Overfit.
In addition, the resolution of 32*32 is not a good training material. It should be increased to 64*64. It will be more textured so that the model can have more features to judge.

---
### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

