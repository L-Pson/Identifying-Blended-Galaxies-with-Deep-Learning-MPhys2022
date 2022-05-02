def createModel(modelName, printSummary=False, plot=False):
    from tensorflow.keras import layers
    import tensorflow as tf

        
    elif modelName.lower() == "custom3":
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, (3,3), padding="same", activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(4, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(8, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        

        
    elif modelName.lower() == "tuned2":
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(222, (3,3), padding="same", activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(196, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(74, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(168, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(89, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(237, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        
        
    elif modelName.lower() == "tuned3":
        
        kernelPower1 = 7
        kernelPower2 = 6
        kernelPower3 = 6
        kernelPower4 = 5
        kernelPower5 = 5
        
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2**kernelPower1, (3,3), padding="same", activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(2**kernelPower2, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(2**kernelPower3, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(2**kernelPower4, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(2**kernelPower5, (3,3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2**(kernelPower5-1), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

        
        
    elif modelName.lower() == "tuned5":
        kernelFactor1 = 3
        kernelFactor2 = 36
        kernelFactor3 = 8
        kernelFactor4 = 46
        kernelFactor5 = 54
        densel=35
        
        model = tf.keras.models.Sequential([
        
        
        tf.keras.layers.Conv2D(8*kernelFactor1, (3,3), padding="same",activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(8*kernelFactor2, (3,3), padding="same",activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(8*kernelFactor3, (3,3), padding="same",activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(8*kernelFactor4, (3,3), padding="same",activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(8*kernelFactor5, (3,3), padding="same",activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(densel*8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    
    if printSummary:
        model.summary()
    if plot:
        keras.utils.plot_model(model, show_shapes=True)
        
    return model
