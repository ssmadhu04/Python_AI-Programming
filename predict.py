from utilities import *

model, class_to_idx = load_checkpoint('checkpoint.pth')

probs, classes = predict(args.image_path, model, args.topk)

print ('Classes: ', classes)
print('Probability: ', probs)
