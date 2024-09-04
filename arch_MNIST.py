import ao_arch as ar


description = "Basic MNIST"
# 1 channel, 28x28=784 neurons, each corresponding to a
# point in MNIST (downsampled to 28x28 bitmap)
arch_i = [28 * 28]
# 1 channel or dimension of output, 4 neurons, corresponding to 2^4=16 binary to code for 0-9 int, the MNIST labels
arch_z = [4]
# 4 c or control neurons are included by default in the 1st channel-- 0-label, 1-force_positive, 2-force_negative, 3-default pleasure instinct triggered when I__flat[0]=1 and Z of previous step Z__flat[0]=1
arch_c = []
# specifies how the neurons are connected;
# in this case, all neurons are connected to all others
connector_function = "rand_conn"
# used 360, 180 before to good success
connector_parameters = [392, 261, 784, 4]
arch = ar.Arch(
    arch_i, arch_z, arch_c, connector_function, connector_parameters, description
)
