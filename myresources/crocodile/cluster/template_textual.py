
@dataclass
class RemoteMachineConfig:
    # conn
    job_id: str = field(default_factory=lambda: tb.randstr(noun=True))
    base_dir: str = f"~/tmp_results/remote_machines/jobs"
    description: str = ""
    ssh_params: dict[str, Union[str, int]] = field(default_factory=lambda: {})
    ssh_obj: Union[tb.SSH, SelfSSH, None] = None

    # data
    copy_repo: bool = False
    update_repo: bool = False
    install_repo: bool = False
    update_essential_repos: bool = True
    data: Optional[list[Any]] = None
    transfer_method: TRANSFER_METHOD = "sftp"
    cloud_name: Optional[str] = None

    # remote machine behaviour
    open_console: bool = True
    notify_upon_completion: bool = False
    to_email: Optional[str] = None
    email_config_name: Optional[str] = None

    # execution behaviour
    launch_method: LAUNCH_METHOD = "remotely"
    kill_on_completion: bool = False
    ipython: bool = False
    interactive: bool = False
    pdb: bool = False
    pudb: bool = False
    wrap_in_try_except: bool = False
    parallelize: bool = False
    lock_resources: bool = True
    max_simulataneous_jobs: int = 1
    workload_params: Optional[WorkloadParams] = None


class MyApp(App):
    async def on_load(self, event):
        self.config = RemoteMachineConfig()

        # Create Input widgets for each parameter
        self.host_input = Input("Host: ", default=self.config.host)
        self.username_input = Input("Username: ", default=self.config.username)
        self.password_input = Input("Password: ", default=self.config.password, password=True)
        self.port_input = Input("Port: ", default=str(self.config.port))

        # Create a Button widget to submit the choices
        self.submit_button = Button("Submit", on_click=self.submit)

        # Add the widgets to the layout
        self.layout = self.host_input | self.username_input | self.password_input | self.port_input | self.submit_button

    async def submit(self, button):
        # Update the config with the user's choices
        self.config.host = self.host_input.value
        self.config.username = self.username_input.value
        self.config.password = self.password_input.value
        self.config.port = int(self.port_input.value)

        # Close the interface
        await self.exit()

if __name__ == "__main__":
    app = MyApp()
    app.run()
    
    # Print the updated config
    print(app.config)
