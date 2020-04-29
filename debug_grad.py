import torch 

def gradient_hook(self, grad_input, grad_output):
    print(f"<<< {self} <<<")
    print(f"Inside {self.__class__.__name__}'s backward")
    
    print(f"Input gradient norm = {grad_input[0].norm()}")
    print(f"Output gradient norm = {grad_output[0].norm()}")
    print("---" * 5)

    if torch.isnan(grad_output[0]).any():
        print("Found NaN values in output")
        print(f"Gradient Norm = {grad_output.norm()}")


def forward_hook(self, input, output):
    print(f">>> {self} forward >>>")
    print(input[0].data)
    print(output.data)
    print("---" * 10)