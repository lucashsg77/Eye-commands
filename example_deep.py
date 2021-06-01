from DeepEyeCommander import *

model = tf.keras.models.load_model('./Models/mark3')
commander = DeepEyeCommander(model=model)
commander.run_demo()