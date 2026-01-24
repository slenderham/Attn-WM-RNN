# Plastic E-I RNN with Attentional Feedback

- Ongoing project. For demonstration purpose only.
- Multi-area excitatory-inhibitory RNN
    - Reward-Modulated Hebbian plasticity in recurrent synapses


$$
\begin{align*}
x_t &= (1-\alpha_x)x_{t-1} + \alpha_x((W^f+W^p_{t-1})h_{t-1} + Uz_t + b) + \sqrt{2\alpha_x}\epsilon_t\\
    h_t &= \sigma(x_t)\\
    W_t^p &= (1-\alpha_W)W_{t-1}^p + r_tA\odot (h_t h_t^T + \sqrt{2\alpha_W}\xi_t)
\end{align*}
$$
