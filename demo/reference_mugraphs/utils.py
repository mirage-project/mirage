import torch
import logging

def analyze_differences(tensor1, tensor2, logger, rtol=1e-5, atol=1e-8, max_examples=20):
    """
    Analyze and output elements that don't satisfy the close condition between two tensors
    
    Parameters:
        tensor1, tensor2: Tensors to compare
        logger: Logger object for output
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_examples: Maximum number of mismatched elements to display
    """
    # Ensure both tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensor shapes don't match"
    
    # Calculate which elements don't satisfy the close condition
    close_mask = torch.isclose(tensor1, tensor2, rtol=rtol, atol=atol)
    not_close_mask = ~close_mask
    
    # Count mismatched elements
    total_not_close = not_close_mask.sum().item()
    total_elements = tensor1.numel()
    percentage = 100 * total_not_close / total_elements
    
    logger.info(f"Total elements: {total_elements}")
    logger.info(f"Mismatched elements: {total_not_close} ({percentage:.4f}%)")
    
    if total_not_close == 0:
        logger.info("All elements satisfy the close condition!")
        return
    
    # Find indices of mismatched elements
    not_close_indices = not_close_mask.nonzero()
    
    # Calculate maximum absolute and relative errors
    abs_diff = (tensor1 - tensor2).abs()
    max_abs_diff = abs_diff.max().item()
    max_abs_diff_index = abs_diff.argmax().item()
    
    rel_diff = abs_diff / (tensor2.abs() + atol)
    max_rel_diff = rel_diff.max().item()
    max_rel_diff_index = rel_diff.argmax().item()
    
    logger.info(f"Maximum absolute error: {max_abs_diff}")
    logger.info(f"Maximum relative error: {max_rel_diff}")
    
    # Get absolute errors for all mismatched elements
    error_data = []
    for i in range(len(not_close_indices)):
        idx = tuple(not_close_indices[i].tolist())
        t1_val = tensor1[idx].item()
        t2_val = tensor2[idx].item()
        abs_err = abs(t1_val - t2_val)
        rel_err = abs_err / (abs(t2_val) + atol)
        error_data.append((idx, t1_val, t2_val, abs_err, rel_err))
    
    # Sort by absolute error in descending order
    error_data.sort(key=lambda x: x[3], reverse=True)
    
    # Output examples of mismatched elements
    logger.info("\nMismatched elements examples (sorted by absolute error):")
    logger.info(f"{'Index':<20} {'Tensor1 Value':<15} {'Tensor2 Value':<15} {'Absolute Error':<15} {'Relative Error':<15}")
    logger.info("-" * 80)
    
    # Limit the number of examples shown
    num_to_show = min(total_not_close, max_examples)
    for i in range(num_to_show):
        idx, t1_val, t2_val, abs_err, rel_err = error_data[i]
        
        # Format output
        idx_str = str(idx)
        logger.info(f"{idx_str:<20} {t1_val:<15.6f} {t2_val:<15.6f} {abs_err:<15.6e} {rel_err:<15.6e}")
    
    if total_not_close > max_examples:
        logger.info(f"\n... {total_not_close - max_examples} more mismatched elements not shown")

    # interleavingly print ench line of two tensors
    logger.info("\nInterleavingly print each line of two tensors:")
    for i in range(len(tensor1)):
        logger.info(f"Line {i}:")
        logger.info(f"{tensor1[i]};\n{tensor2[i]}")
