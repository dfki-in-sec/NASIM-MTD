import os
import yaml
import os.path as osp


SCENARIO_DIR = osp.dirname(osp.abspath(__file__))

# default subnet address for internet
INTERNET = 0

# Constants
NUM_ACCESS_LEVELS = 2
NO_ACCESS = 0
USER_ACCESS = 1
ROOT_ACCESS = 2
DEFAULT_HOST_VALUE = 0

# scenario property keys
SUBNETS = "subnets"
TOPOLOGY = "topology"
SENSITIVE_HOSTS = "sensitive_hosts"
HONEYPOTS = "honeypots" #MT-T2
SERVICES = "services"
VUL = "vul" #CBP
OS = "os"
PROCESSES = "processes"
EXPLOITS = "exploits"
PRIVESCS = "privilege_escalation"
SERVICE_SCAN_COST = "service_scan_cost"
WIRETAPPING_COST = "wiretapping_cost" #CBP3
VUL_SCAN_COST = "vul_scan_cost" #CBP
OS_SCAN_COST = "os_scan_cost"
SUBNET_SCAN_COST = "subnet_scan_cost"
PROCESS_SCAN_COST = "process_scan_cost"
HOST_CONFIGS = "host_configurations"
FIREWALL = "firewall"
HOSTS = "host"
STEP_LIMIT = "step_limit"
MOVEMENT_TIME = "movement_time" #MT-T
ACCESS_LEVELS = "access_levels"

# scenario exploit keys
EXPLOIT_SERVICE = "service"
EXPLOIT_OS = "os"
EXPLOIT_VUL = "vul"
EXPLOIT_PROB = "prob"
EXPLOIT_COST = "cost"
EXPLOIT_ACCESS = "access"
EXPLOIT_CREDENTIALS_NEEDED = "credentials_needed" #CBP3

# scenario privilege escalation keys
PRIVESC_PROCESS = "process"
PRIVESC_OS = "os"
PRIVESC_PROB = "prob"
PRIVESC_COST = "cost"
PRIVESC_ACCESS = "access"
PRIVESC_CREDENTIALS_TOFIND = "credentials_tofind" #CBP4

# host configuration keys
HOST_SERVICES = "services"
HOST_VUL = "vul" #CBP
HOST_PROCESSES = "processes"
HOST_OS = "os"
HOST_FIREWALL = "firewall"
HOST_VALUE = "value"
HOST_CREDENTIALS_NEEDED = "credentials_needed" #CBP3
HOST_CREDENTIALS_TOFIND = "credentials_tofind" #CBP3

def load_yaml(file_path):
    """Load yaml file located at file path.

    Parameters
    ----------
    file_path : str
        path to yaml file

    Returns
    -------
    dict
        contents of yaml file

    Raises
    ------
    Exception
        if theres an issue loading file. """
    with open(file_path) as fin:
        content = yaml.load(fin, Loader=yaml.FullLoader)
    return content


def get_file_name(file_path):
    """Extracts the file or dir name from file path

    Parameters
    ----------
    file_path : str
        file path

    Returns
    -------
    str
        file name with any path and extensions removed
    """
    full_file_name = file_path.split(os.sep)[-1]
    file_name = full_file_name.split(".")[0]
    return file_name
