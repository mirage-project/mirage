from cuda.bindings import driver, nvrtc


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

checkCudaErrors(driver.cuInit(0))

def _queryMulticastSupport(cu_device):
    # Query multicast object support
    multicast_supported = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        cu_device
    ))
    return bool(multicast_supported)

def queryMulticastSupport(device_id):
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    return _queryMulticastSupport(cu_device)

def _queryVMMsupport(cu_device):
    # Query virtual memory management support
    vmm_supported = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        cu_device
    ))
    return bool(vmm_supported)

def queryVMMsupport(device_id):
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    return _queryVMMsupport(cu_device)

def _queryPeerAccessSupported(cu_device_from, cu_device_to):
    # Query peer access support
    peer_access_supported = checkCudaErrors(driver.cuDeviceCanAccessPeer(
        cu_device_from,
        cu_device_to
    ))
    return bool(peer_access_supported)

def queryPeerAccessSupported(device_id_from, device_id_to):
    cu_device_from = checkCudaErrors(driver.cuDeviceGet(device_id_from))
    cu_device_to = checkCudaErrors(driver.cuDeviceGet(device_id_to))
    return _queryPeerAccessSupported(cu_device_from, cu_device_to)

def _queryHandleTypePosixFileDescriptorSupported(cu_device):
    # Query POSIX shared memory support
    supported = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        cu_device
    ))
    return bool(supported)

def queryHandleTypePosixFileDescriptorSupported(device_id):
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    return _queryHandleTypePosixFileDescriptorSupported(cu_device)

def queryComputeCapability(device_id):
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    major = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cu_device
    ))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cu_device
    ))
    return (major, minor)