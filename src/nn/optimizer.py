"""
Optimizers depend on the model that you are going with:
    Assume that our dataset has been loaded, normalized, split, the model has been defined, 
    and then we have a defined function as well. This is just a model for the simplest kind of
    machine learning--linear regression--and we can always update as necessary.
"""
import numpy as np

class Optimizer:
    def stochastic_gradient_descent(
        self,
        x, y,
        epochs=np.random.randint(50, 100),
        learning_rate=0.01,  # or 0.001
        batch_size=16,       # can also be 32, 64
        stopping_threshold=1e-6,
    ):
        m = np.random.randn()
        b = np.random.randn()
        n = len(x)  # The number of data points
        previous_loss = np.inf

        for i in range(epochs):
            indices = np.random.permutation(n)
            x = x[indices]
            y = y[indices]
            for j in range(0, n, batch_size):
                x_batch = x[j:j + batch_size]
                y_batch = y[j:j + batch_size]

                # Make predictions with current m, b
                y_pred = model(m, x_batch, b)

                # Compute the gradients
                m_gradient = 2 * np.mean(x_batch * (y_batch - y_pred))
                b_gradient = 2 * np.mean(y_batch - y_pred)

                # Gradient clipping which prevents exploding gradient problem
                clip_value = 1.0
                m_gradient = np.clip(m_gradient, -clip_value, clip_value)
                b_gradient = np.clip(b_gradient, -clip_value, clip_value)

                # Update the model parameters
                m -= learning_rate * m_gradient
                b -= learning_rate * b_gradient

            # Compute the epoch loss
            y_pred = model(m, x, b)
            current_loss = loss(y, y_pred)

            # Check against the stopping threshold
            if abs(previous_loss - current_loss) < stopping_threshold:
                break

            previous_loss = current_loss

        return m, b

    def RMSProp(
        self,
        x, y,
        epochs=100,
        learning_rate=0.01,
        batch_size=32,
        stopping_threshold=1e-6,
        beta=0.9, #decay parameter
        epsilon=1e-8 #prevents division by zero when making updates,
    ):
        # Initialize the model parameters randomly
        m = np.random.randn()
        b = np.random.randn()
        # Initialize accumulators for squared gradients
        s_m = 0
        s_b = 0
        n = len(x)
        previous_loss = np.inf
        for i in range(epochs):
            indices = np.random.permutation(n)
            x = x[indices]
            y = y[indices]
            for j in range(0, n, batch_size):
                x_batch = x[j:j + batch_size]
                y_batch = y[j:j + batch_size]

                # Make predictions with current m, b
                y_pred = model(m, x_batch, b)

                # Compute the gradients
                m_gradient = 2 * np.mean(x_batch * (y_batch - y_pred))
                b_gradient = 2 * np.mean(y_batch - y_pred)
                
                # Update accumulators
                s_m = beta * s_m + (1 - beta) * (m_gradient**2)
                s_b = beta * s_b + (1 - beta) * (b_gradient**2)
                # Update parameters
                m -= learning_rate * m_gradient / (np.sqrt(s_m) + epsilon)
                b -= learning_rate * b_gradient / (np.sqrt(s_b) + epsilon)
            
            y_pred = model(m, x, b)
            current_loss = loss(y, y_pred)

            # Check against the stopping threshold
            if abs(previous_loss - current_loss) < stopping_threshold:
                break

            previous_loss = current_loss

        return m, b

    def adam_optimization(
        self,
        x,
        y,
        epochs=np.random.randint(50, 100),
        learning_rate=0.01,
        batch_size=16,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        # Initialize the model parameters
        m = np.random.randn()
        b = np.random.randn()
        # Initialize first and second moment vectors
        m_m, v_m = 0, 0
        m_b, v_b = 0, 0
        n = len(x)
        previous_loss = np.inf
        t = 0  # Initialize timestep
