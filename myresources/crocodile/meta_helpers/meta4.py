from crocodile.core import randstr, str2timedelta, Struct
from crocodile.meta_helpers.meta1 import Log
import time
import datetime as dtm
from datetime import datetime
from typing import Union, Any, Optional, Callable


class Scheduler:
    def __init__(self, routine: Callable[['Scheduler'], Any], wait: Union[str, int, float] = "2m", max_cycles: int = 10000000000,
                 exception_handler: Optional[Callable[[Union[Exception, KeyboardInterrupt], str, 'Scheduler'], Any]] = None,
                 logger: Optional[Log] = None, sess_stats: Optional[Callable[['Scheduler'], dict[str, Any]]] = None, records: Optional[list[list[Any]]] = None):
        self.routine = routine  # main routine to be repeated every `wait` time period
        self.wait_sec = str2timedelta(shift=wait).total_seconds() if isinstance(wait, str) else wait  # wait period between routine cycles.
        self.logger = logger if logger is not None else Log(name="SchedLogger_" + randstr(noun=True))
        self.exception_handler = exception_handler if exception_handler is not None else self.default_exception_handler
        self.sess_start_time = datetime.now()  # to be reset at .run
        self.records: list[list[Any]] = records if records is not None else []
        self.cycle: int = 0
        self.max_cycles: int = max_cycles
        self.sess_stats = sess_stats or (lambda _sched: {})
    def __repr__(self): return f"Scheduler with {self.cycle} cycles ran so far. Last cycle was at {self.sess_start_time}."
    def run(self, max_cycles: Optional[int] = None, until: str = "2050-01-01", starting_cycle: Optional[int] = None):
        if starting_cycle is not None: self.cycle = starting_cycle
        self.max_cycles, self.sess_start_time = max_cycles or self.max_cycles, datetime.now()
        while datetime.now() < datetime.fromisoformat(until) and self.cycle < self.max_cycles:  # 1- Time before Ops, and Opening Message
            time1 = datetime.now()
            self.logger.info(f"🚦Cycle {str(self.cycle).zfill(6)}. Total Run⏳ = {str(datetime.now() - self.sess_start_time).split('.', maxsplit=1)[0]}. UTC🕜 {datetime.now(tz=dtm.UTC).strftime('%d %H:%M:%S')}")
            try: self.routine(self)
            except Exception as ex: self.exception_handler(ex, "routine", self)  # 2- Perform logic
            time_left = int(self.wait_sec - (datetime.now() - time1).total_seconds())  # 4- Conclude Message
            self.cycle += 1
            self.logger.info(f"🏁Cycle {str(self.cycle - 1).zfill(6)} in {str(datetime.now() - time1).split('.', maxsplit=1)[0]}. Sleeping for {self.wait_sec}s ({time_left}s left)\n" + "-" * 100)
            try: time.sleep(time_left if time_left > 0 else 0.1)  # # 5- Sleep. consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex:
                self.exception_handler(ex, "sleep", self)
                return  # that's probably the only kind of exception that can rise during sleep.
        self.record_session_end(reason=f"Reached maximum number of cycles ({self.max_cycles})" if self.cycle >= self.max_cycles else f"Reached due stop time ({until})")
    def get_records_df(self):
        import pandas as pd
        return pd.DataFrame.from_records(self.records, columns=["start", "finish", "duration", "cycles", "termination reason", "logfile"] + list(self.sess_stats(self).keys()))
    def record_session_end(self, reason: str = "Not passed to function."):
        end_time = datetime.now()
        duration = end_time - self.sess_start_time
        sess_stats = self.sess_stats(self)
        self.records.append([self.sess_start_time, end_time, duration, self.cycle, reason, self.logger.file_path] + list(sess_stats.values()))
        summ = {
            "start time": f"{str(self.sess_start_time)}",
            "finish time": f"{str(end_time)}.",
            "duration": f"{str(duration)} | wait time {self.wait_sec}s",
            "cycles ran": f"{self.cycle} | Lifetime cycles = {self.get_records_df()['cycles'].sum()}",
            "termination reason": reason, "logfile": self.logger.file_path
                }
        tmp = Struct(summ).update(sess_stats).print(as_config=True, return_str=True, quotes=False)
        assert isinstance(tmp, str)
        self.logger.critical("\n--> Scheduler has finished running a session. \n" + tmp + "\n" + "-" * 100)
        df = self.get_records_df()
        df["start"] = df["start"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["finish"] = df["finish"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["duration"] = df["duration"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        self.logger.critical("\n--> Logger history.\n" + str(df))
        return self
    def default_exception_handler(self, ex: Union[Exception, KeyboardInterrupt], during: str, sched: 'Scheduler') -> None:  # user decides on handling and continue, terminate, save checkpoint, etc.  # Use signal library.
        print(sched)
        self.record_session_end(reason=f"during {during}, " + str(ex))
        self.logger.exception(ex)
        raise ex


class SchedulerV2:
    def __init__(self, routine: Callable[['SchedulerV2'], Any],
                 wait_ms: int,
                 exception_handler: Optional[Callable[[Union[Exception, KeyboardInterrupt], str, 'SchedulerV2'], Any]] = None,
                 logger: Optional[Log] = None,
                 sess_stats: Optional[Callable[['SchedulerV2'], dict[str, Any]]] = None,
                 records: Optional[list[list[Any]]] = None):
        self.routine = routine  # main routine to be repeated every `wait` time period
        self.logger = logger if logger is not None else Log(name="SchedLogger_" + randstr(noun=True))
        self.exception_handler = exception_handler if exception_handler is not None else self.default_exception_handler
        self.records: list[list[Any]] = records if records is not None else []
        self.wait_ms = wait_ms  # wait period between routine cycles.
        self.cycle: int = 0
        self.max_cycles: int
        self.sess_start_time: int
        self.sess_stats = sess_stats or (lambda _sched: {})
    def __repr__(self): return f"Scheduler with {self.cycle} cycles ran so far. Last cycle was at {self.sess_start_time}."
    def run(self, max_cycles: Optional[int], until_ms: int):
        if max_cycles is not None:
            self.max_cycles = max_cycles
        self.sess_start_time = time.time_ns() // 1_000_000
        while (time.time_ns() // 1_000_000) < until_ms and self.cycle < self.max_cycles:
            # 1- Time before Ops, and Opening Message
            time1 = time.time_ns() // 1_000_000
            self.logger.info(f"Starting Cycle {str(self.cycle).zfill(5)}. Total Run Time = {str(time1 - self.sess_start_time).split('.', maxsplit=1)[0]}. UTC🕜 {datetime.now(tz=dtm.UTC).strftime('%d %H:%M:%S')}")
            try:
                self.routine(self)
            except Exception as ex:
                self.exception_handler(ex, "routine", self)  # 2- Perform logic
            time2 = time.time_ns() // 1_000_000
            time_left = int(self.wait_ms - (time2 - time1))  # 4- Conclude Message
            self.cycle += 1
            self.logger.info(f"Finishing Cycle {str(self.cycle - 1).zfill(5)} in {str(time2 - time1).split('.', maxsplit=1)[0]}. Sleeping for {self.wait_ms}ms ({time_left}s left)\n" + "-" * 100)
            try: time.sleep(time_left if time_left > 0 else 0.1)  # # 5- Sleep. consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex:
                self.exception_handler(ex, "sleep", self)
                return  # that's probably the only kind of exception that can rise during sleep.
        self.record_session_end(reason=f"Reached maximum number of cycles ({self.max_cycles})" if self.cycle >= self.max_cycles else f"Reached due stop time ({until_ms})")
    def get_records_df(self):
        import pandas as pd
        return pd.DataFrame.from_records(self.records, columns=["start", "finish", "duration", "cycles", "termination reason", "logfile"] + list(self.sess_stats(self).keys()))
    def record_session_end(self, reason: str):
        end_time = time.time_ns() // 1_000_000
        duration = end_time - self.sess_start_time
        sess_stats = self.sess_stats(self)
        self.records.append([self.sess_start_time, end_time, duration, self.cycle, reason, self.logger.file_path] + list(sess_stats.values()))
        summ = {"start time": f"{str(self.sess_start_time)}",
                "finish time": f"{str(end_time)}.",
                "duration": f"{str(duration)} | wait time {self.wait_ms / 1_000: 0.1f}s",
                "cycles ran": f"{self.cycle} | Lifetime cycles = {self.get_records_df()['cycles'].sum()}",
                "termination reason": reason, "logfile": self.logger.file_path
                }
        tmp = Struct(summ).update(sess_stats).print(as_config=True, return_str=True, quotes=False)
        assert isinstance(tmp, str)
        self.logger.critical("\n--> Scheduler has finished running a session. \n" + tmp + "\n" + "-" * 100)
        df = self.get_records_df()
        df["start"] = df["start"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["finish"] = df["finish"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["duration"] = df["duration"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        self.logger.critical("\n--> Logger history.\n" + str(df))
        return self
    def default_exception_handler(self, ex: Union[Exception, KeyboardInterrupt], during: str, sched: 'SchedulerV2') -> None:  # user decides on handling and continue, terminate, save checkpoint, etc.  # Use signal library.
        print(sched)
        self.record_session_end(reason=f"during {during}, " + str(ex))
        self.logger.exception(ex)
        raise ex
