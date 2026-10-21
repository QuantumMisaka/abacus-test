import os,sys,glob,re,traceback,math
from ..resultAbacus import ResultAbacus
from .. import comm
import numpy as np

KS_SOLVER_LIST = ['DA','DS','GE','GV','BP','CG','CU','PE','LA']
_CONFLICT = object()


def _ion_step(line):
    match = re.search(r"\bION\s*(?:STEP\s*)?[:=]?\s*(\d+)\b", line, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _current_segment(lines):
    """Keep the latest identifiable ionic run in an appended log."""
    last_step = None
    start = 0
    for index, line in enumerate(lines):
        step = _ion_step(line)
        if step is None:
            continue
        if last_step is not None and step < last_step:
            start = index
        last_step = step
    return lines[start:]


def _discover_stdout_paths(result):
    """Find named files for this run; use arbitrary root files only as a last resort."""
    paths = []

    def add(path):
        if path is None:
            return
        path = os.fspath(path)
        if not os.path.isabs(path) and not os.path.isfile(path):
            path = os.path.join(os.fspath(result.PATH), path)
        path = os.path.abspath(path)
        if os.path.isfile(path) and path not in paths:
            paths.append(path)

    # FindOutput scans every file for a banner and may return an old log. Only
    # the caller's explicit output path carries that provenance by itself.
    if getattr(result, "_OUTPUTf_explicit", False):
        add(getattr(result, "OUTPUTf", None))
    for name in ("abacus.log", "abacus.out", "out.log"):
        add(os.path.join(result.PATH, name))
    add(getattr(result, "LOGf", None))
    if not paths:
        candidates = []
        # Unknown *.log files may belong to an earlier calculation. An
        # alternate log name must be supplied through OUTPUTf explicitly.
        for pattern in ("*.out", "*.stdout"):
            candidates.extend(glob.glob(os.path.join(os.fspath(result.PATH), pattern)))
        for path in sorted(set(candidates)):
            add(path)
    return paths


def _stdout_lines(result, evidence, label):
    """Select one signal from one current source, never concatenate log files.

    A named screen output and the current running log may carry different
    signals. If two sources carry conflicting values for the same signal,
    neither can be used as a factual relaxation result.
    """
    selected_lines = None
    selected_value = None
    for path in _discover_stdout_paths(result):
        lines = _current_segment(comm.ReadFile(path, warn=False))
        value = evidence(lines)
        if value is _CONFLICT:
            print("Warning: conflicting %s within %s" % (label, path))
            return [], True
        if value is None:
            continue
        if selected_lines is not None and value != selected_value:
            print("Warning: conflicting %s across current ABACUS log sources" % label)
            return [], True
        selected_lines, selected_value = lines, value
    return selected_lines or [], False


def _gradient_markers(lines):
    """Group aliases by explicit ionic step; ambiguous unlabeled aliases are unknown."""
    number = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    force, stress = [], []
    force_steps, stress_steps = {}, {}
    step = None
    saw_unlabeled_kind = None
    for line in lines:
        next_step = _ion_step(line)
        if next_step is not None:
            if (step is None and (force or stress)) or (step is not None and next_step < step):
                # An ionic-step restart begins a new run segment. A labeled
                # segment also supersedes earlier unlabeled observations.
                force, stress = [], []
                force_steps, stress_steps = {}, {}
                saw_unlabeled_kind = None
            step = next_step
        kind, value = None, None
        if re.search(r"largest\s+gradient\s+in\s+stress", line, re.IGNORECASE):
            match = re.findall(number, line)
            if match:
                kind, value = "stress", float(match[-1])
        elif re.search(r"largest\s+gradient\s+in\s+force", line, re.IGNORECASE):
            match = re.findall(number, line)
            if match:
                kind, value = "gradient_force", float(match[-1])
        elif re.search(r"largest\s+gradient\s+is", line, re.IGNORECASE):
            match = re.findall(number, line)
            if match:
                kind, value = "gradient_is", float(match[-1])
        elif re.search(r"largest\s+grad(?:ient)?\s*(?:\(\s*eV\s*/\s*(?:A|Angstrom|Å)\s*\))?\s*[:=]", line, re.IGNORECASE):
            match = re.findall(number, re.split(r"[:=]", line, maxsplit=1)[1])
            if match:
                kind, value = "gradient_unit", float(match[0])
        elif re.search(r"largest\s+force(?:\s*=|\s+)", line, re.IGNORECASE):
            match = re.findall(number, re.split(r"force", line, maxsplit=1, flags=re.IGNORECASE)[-1])
            if match:
                kind, value = "force", float(match[0])
        if value is None:
            continue
        values = stress if kind == "stress" else force
        steps = stress_steps if kind == "stress" else force_steps
        if step is None and kind != "stress":
            if saw_unlabeled_kind is not None and kind != saw_unlabeled_kind:
                return _CONFLICT
            saw_unlabeled_kind = kind
        if step is not None and step in steps:
            if not math.isclose(values[steps[step]], value, rel_tol=1e-5, abs_tol=1e-6):
                return _CONFLICT
            continue
        if step is not None:
            steps[step] = len(values)
        values.append(value)
    return (force or None, stress or None)


def _gradient_component(lines, index):
    gradients = _gradient_markers(lines)
    return _CONFLICT if gradients is _CONFLICT else gradients[index]


def _relax_status(lines):
    for line in reversed(lines):
        if "Relaxation is converged!" in line or "Lattice relaxation is converged!" in line or "Ion relaxation is converged!" in line:
            return True
        if ("Relaxation is not converged yet!" in line or "Ion relaxation is not converged yet" in line or
                "Lattice relaxation is not converged yet" in line):
            return False
    return None


def _mobility_components(result):
    """Read per-atom Cartesian mobility flags from the root STRU."""
    mobility = []
    in_positions = False
    float_pattern = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
    value_counts = {
        "v": (3,), "vel": (3,), "velocity": (3,),
        "mag": (1, 3), "magmom": (1, 3),
        "angle1": (1,), "angle2": (1,),
        "constrain": (1, 3), "sc": (1, 3), "lambda": (1, 3),
    }

    def parse_flags(values):
        if len(values) != 3:
            return None
        try:
            flags = [int(value) for value in values]
        except ValueError:
            return None
        return flags if all(value in (0, 1) for value in flags) else None

    def parse_position(tokens):
        flags = None
        index = 3
        while index < len(tokens):
            tag = tokens[index].lower()
            if tag == "m":
                if flags is not None:
                    return None
                flags = parse_flags(tokens[index + 1:index + 4])
                if flags is None:
                    return None
                index += 4
            elif flags is None and index == 3 and tag not in value_counts:
                flags = parse_flags(tokens[index:index + 3])
                if flags is None:
                    return None
                index += 3
            elif tag in value_counts:
                end = index + 1
                while end < len(tokens) and tokens[end].lower() not in value_counts and tokens[end].lower() != "m":
                    end += 1
                values = tokens[index + 1:end]
                if len(values) not in value_counts[tag] or not all(float_pattern.match(value) for value in values):
                    return None
                index = end
            else:
                return None
        return flags

    for line in result.STRU:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("ATOMIC_POSITIONS"):
            in_positions = True
            continue
        if not in_positions:
            continue
        tokens = stripped.split("#", 1)[0].split()
        if len(tokens) < 3 or not all(float_pattern.match(token) for token in tokens[:3]):
            continue
        mobility.append(parse_position(tokens))
    return mobility


def _energy_difference(lines):
    number = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    for line in reversed(lines):
        if re.search(r"energy\s+difference|energy\s+diff", line, re.IGNORECASE):
            values = re.findall(number, line)
            if values:
                return float(values[-1])
    return None


def _force_tables(lines):
    forces = []
    value_pattern = re.compile(r'^\s*\S+\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$')
    for i, line in enumerate(lines):
        if 'TOTAL-FORCE (eV/Angstrom)' not in line:
            continue
        j = i + 1
        while j < min(i + 10, len(lines)) and not value_pattern.match(lines[j]):
            j += 1
        if j >= min(i + 10, len(lines)):
            continue
        force = []
        while j < len(lines) and value_pattern.match(lines[j]):
            force.extend(float(value) for value in lines[j].split()[1:4])
            j += 1
        if force:
            forces.append(force)
    return forces or None

class Abacus(ResultAbacus):
    
    @ResultAbacus.register(version="the version of ABACUS")
    def GetVersion(self):
        version = "unknown"
        commit = "unknown"
        if self.JSON:
            version = self.JSON.get("general_info",{}).get("version","unknown")
            commit = self.JSON.get("general_info",{}).get("commit","unknown")
        else:
            if len(self.LOG) > 0:
                for line_idx,line in enumerate(self.LOG):
                    if "WELCOME TO ABACUS" in line:
                        version = line.split()[-1]
                        if version[0].lower() != 'v':
                            print("Unknow version of '%s'" % version)
                        self['version'] = version
                        return
                    elif line[30:36] == "ABACUS":
                        version = line[36:].strip()
                        commit = "unknown"
                        for ii in range(30):
                            if line_idx + ii >= len(self.LOG):
                                break
                            if "Commit:" in self.LOG[line_idx + ii]:
                                commit = re.split(":",self.LOG[line_idx + ii].strip(),maxsplit=1)[-1].strip()
                                break
        self['version'] = version + "(" + commit + ")"
        return
                                              
    @ResultAbacus.register(ncore="the mpi cores",
                           omp_num="the omp cores",)
    def GetNcore(self):
        ncore = omp_num = None
        if self.JSON:
            ncore,_tmp = comm.get_abacus_json(self.JSON,["general_info","mpi_num"])
            omp_num,_tmp = comm.get_abacus_json(self.JSON,["general_info","omp_num"])
        elif self.LOG:    
            for line in self.LOG:
                if "DSIZE =" in line:
                    ncore = int(line.split()[-1])
                    break
                
        self["ncore"] = ncore
        self["omp_num"] = omp_num
    
    @ResultAbacus.register(normal_end="if the job is normal ending")
    def GetNormalEnd(self):
        if len(self.LOG) == 0:
            self['normal_end'] = None
            return
        elif len(self.LOG) > 0 and " Total  Time  :" in self.LOG[-1]:
            self['normal_end'] = True
            return
        else:
            self['normal_end'] = False

            print("Job is not normal ending!!! The latest 10 lines are:")
            if len(self.LOG) < 10:
                print(''.join(self.LOG))
            else:
                print(''.join(self.LOG[-10:]))

    @ResultAbacus.register(INPUT="a dict to store the setting in OUT.xxx/INPUT")
    def GetInputParameter(self):
        def str2intfloat(ii):
            try:
                return int(ii)
            except:
                pass
            try:
                return float(ii)
            except:
                return ii

        # The input at the job root is the user-owned source of truth.  The
        # copy in OUT.* is generated runtime state and may contain defaults or
        # values from a different execution, so it must not replace self.INPUT.
        input_context = self.INPUT
        
        readinput = False
        INPUT = {}
        for i,iline in enumerate(input_context):
            if iline.strip() == 'INPUT_PARAMETERS':
                readinput = True
            elif iline.strip() == '' or iline.strip()[0] in ['#']:
                continue
            elif readinput:
                sline = re.split('[ \t]',iline.split("#")[0].strip(),maxsplit=1)
                if len(sline) == 2:
                    INPUT[sline[0].lower()] = str2intfloat(sline[1].strip())
        self["INPUT"] = INPUT
        return INPUT

    @ResultAbacus.register(kpt="list, the K POINTS setting in KPT file")
    def GetKptParam(self):
        if len(self.KPT) > 3:
            self["kpt"] = [int(i) for i in self.KPT[3].split()[:3]]
        else:
            self["kpt"] = None

    @ResultAbacus.register(fft_grid = "fft grid for charge/potential")
    def GetGridInfo(self):
        fft_grid = None
        for i,line in enumerate(self.LOG):
            if "fft grid for charge/potential =" in line:
                fft_grid = [float(i.strip()) for i in line.split('=')[1].replace("[","").replace("]","").split(',')]
                break     
        self["fft_grid"] = fft_grid
    
    @ResultAbacus.register(nbase = "number of basis in LCAO")
    def GetLogParamNBase(self): 
        nbase = None
        for i,line in enumerate(self.LOG):
            if "NLOCAL =" in line:
                nbase = int(line.split()[2])
                break
        self['nbase'] = nbase

    @ResultAbacus.register(noccu_band="number of occupied bands")
    def GetNumOccuBand(self):
        noccu_band = None
        for i,line in enumerate(self.LOG):
            if "occupied bands" in line:
                noccu_band = int(line.split()[3])
                break
        self['noccu_band'] = noccu_band

    @ResultAbacus.register(nbands="number of bands",
                           nkstot = "total K point number",
                           ibzk = "irreducible K point number",
                           natom ="total atom number",
                           nelec = "total electron number",
                           nelec_dict = "dict of electron number of each species",
                           point_group="point group",
                           point_group_in_space_group="point group in space group")
    def GetLogParam(self): 
        if self.JSON:
            self["nbands"],_ = comm.get_abacus_json(self.JSON,["init","nband"])
            self["nkstot"],_ = comm.get_abacus_json(self.JSON,["init","nkstot"])
            self["ibzk"],_ = comm.get_abacus_json(self.JSON,["init","nkstot_ibz"])
            self["natom"],_ = comm.get_abacus_json(self.JSON,["init","natom"])
            self["nelec"],_ = comm.get_abacus_json(self.JSON,["init","nelectron"])
            self["nelec_dict"],_ = comm.get_abacus_json(self.JSON,["init","nelectron_each_type"])
            self["fft_grid"],_ = comm.get_abacus_json(self.JSON,["init","fft_grid"])
            self["point_group"],_ = comm.get_abacus_json(self.JSON,["init","point_group"])
            self["point_group_in_space_group"],_ = comm.get_abacus_json(self.JSON,["init","point_group_in_space"])
        else:      
            natom = 0
            nelec = 0
            elec_dict = {}
            point_group = None
            point_group_in_space_group = None
            for i,line in enumerate(self.LOG):
                if "NBANDS =" in line:
                    self['nbands'] = int(line.split()[2])
                elif 'nkstot =' in line:
                    self['nkstot'] = int(line.split()[-1])
                elif 'nkstot_ibz =' in line:
                    self['ibzk'] = int(line.split()[-1])
                elif "            electron number of element" in line:
                    elec_dict[line.split()[4]] = float(line.split()[-1])
                elif 'number of atom for this type =' in line:
                    natom += int(line.split()[-1])
                elif 'total electron number of element' in line:
                    nelec += float(line.split()[-1])
                elif "POINT GROUP =" in line:
                    point_group = line.split("=")[1].strip()
                elif "POINT GROUP IN SPACE GROUP =" in line:
                    point_group_in_space_group = line.split("=")[1].strip()

            self["point_group"] = point_group
            self["point_group_in_space_group"] = point_group_in_space_group
            if natom > 0:
                self["natom"] = natom 
            if nelec > 0:
                self["nelec"] = nelec 
            else:
                self["nelec"] = None

            if elec_dict:
                self["nelec_dict"] = elec_dict
            else:
                self["nelec_dict"] = None

    @ResultAbacus.register(total_mag="total magnetism (Bohr mag/cell)",
                           absolute_mag="absolute magnetism (Bohr mag/cell)",
                           total_mags="total magnetism (Bohr mag/cell) of each ION step",
                           absolute_mags="absolute magnetism (Bohr mag/cell) of each ION step",
                           )
    def GetMagResult(self): 
        total_mags = []
        absolute_mags = []
        tot_mag = None
        abs_mag = None
        ion_step = 1
        for i,line in enumerate(self.LOG):
            if "ION=" in line:
                current_step = int(line.split()[4])
                if current_step > ion_step:
                    if tot_mag is not None:
                        total_mags.append(tot_mag)
                        absolute_mags.append(abs_mag)
                ion_step = current_step 
            elif 'total magnetism (Bohr mag/cell)' in line:
                sline = line.split()
                # if the last three are float, then it is the noncollinear case, else it is collinear case
                try:
                    tot_mag = [float(imag) for imag in sline[-3:]]
                except:
                    tot_mag = float(sline[-1])
            elif 'absolute magnetism' in line:
                abs_mag = float(line.split()[-1])
        
        if tot_mag is not None:    
            self['total_mag'] = tot_mag
            self['absolute_mag'] = abs_mag
            self['total_mags'] = total_mags + [tot_mag]
            self['absolute_mags'] = absolute_mags + [abs_mag]
        else:
            self['total_mag'] = None
            self['absolute_mag'] = None
            self['total_mags'] = None
            self['absolute_mags'] = None
        
    
    @ResultAbacus.register(converge="if the SCF is converged",
                           energy = "the total energy (eV)",
                           energy_ks = "the E_KohnSham, unit in eV",
                           energies = "list of total energy of each ION step",
                           volume = "the volume of cell, in A^3",
                           efermi = "the fermi energy (eV). If has set nupdown, this will be a list of two values. The first is up, the second is down.",
                           energy_per_atom="the total energy divided by natom, (eV)")
    def GetLogResult(self):     
        if self.JSON and self.JSON.get("output") and len(self.JSON.get("output")) > 0:
            output_final = self.JSON.get("output")[-1]
            self["converge"] = output_final.get("scf_converge",None)
            self["energy"] = output_final.get("energy",None)
            self["energies"] = [i.get("energy",None) for i in self.JSON.get("output")]
            self["efermi"] = output_final.get("e_fermi",None)
            cell  = output_final.get("cell",None)
            if cell:
                self["volume"] = abs(np.linalg.det(cell))
            else:
                self["volume"] = None
            if self["natom"] != None and self['energy'] != None:
                self["energy_per_atom"] = self['energy']/self["natom"]
            else:
                self["energy_per_atom"] = None
        else: 
            efermi = None
            converge = None
            energy = None
            energies = []
            energies_ks = []
            volume = None
            for i,line in enumerate(self.LOG):
                if 'charge density convergence is achieved' in line:
                    converge = True
                elif 'convergence has NOT been achieved!' in line or\
                    'convergence has not been achieved' in line:
                    converge = False
                elif "!FINAL_ETOT_IS" in line:
                    energy = float(line.split()[1])
                elif "final etot is" in line:
                    energies.append(float(line.split()[-2]))
                elif "E_KohnSham" in line:
                    energies_ks.append(float(line.split()[-1]))
                elif "Volume (A^3) =" in line:
                    volume = float(line.split()[-1])
                elif 'E_Fermi' in line:
                    if 'E_Fermi_up' in line:
                        if efermi == None:
                            efermi = [None,None]
                        efermi[0] = float(line.split()[-1])
                    elif 'E_Fermi_dw' in line:
                        if efermi == None:
                            efermi = [None,None]
                        efermi[1] = float(line.split()[-1])
                    else:
                        efermi = float(line.split()[-1])

            self["energy"] = energy
            self["converge"] = converge
            self["volume"] = volume
            if energies:
                self["energies"] = energies
            else:
                self["energies"] = None

            if energies_ks:
                self["energy_ks"] = energies_ks[-1]
            else:
                self["energy_ks"] = None

            if efermi != None and isinstance(efermi,list):
                if abs(efermi[0] - efermi[1]) < 1e-6:
                    efermi = efermi[0]
            self["efermi"] = efermi

            if self["natom"] != None and self['energy'] != None:
                self["energy_per_atom"] = self['energy']/self["natom"]
    
    @ResultAbacus.register(force="list[3*natoms], force of the system, if is MD or RELAX calculation, this is the last one",
                           forces = "list of force, the force of each ION step. Dimension is [nstep,3*natom]")
    def GetForceFromLog(self):
        evidence, conflict = _stdout_lines(self, _force_tables, "force tables")
        forces = None if conflict else _force_tables(evidence)
        if forces:        
            self['force'] = forces[-1]
            self["forces"] = forces
        else:
            self['force'] = None
            self["forces"] = None
    
    @ResultAbacus.register(stress="list[9], stress of the system, if is MD or RELAX calculation, this is the last one",
                           virial="list[9], virial of the system,  = stress * volume, and is the last one.",
                           pressure="the pressure of the system, unit in kbar.",
                           stresses="list of stress, the stress of each ION step. Dimension is [nstep,9]",
                           virials="list of virial, the virial of each ION step. Dimension is [nstep,9]",
                           pressures="list of pressure, the pressure of each ION step.")
    def GetStessFromLog(self):
        stresses = []
        for i in range(len(self.LOG)):
            #i = -1*i - 1
            line = self.LOG[i]
            if 'TOTAL-STRESS (KBAR)' in line:
                value_pattern = re.compile(r'^\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$')
                j = i 
                nostress = False
                # find the first line of stress
                while not value_pattern.match(self.LOG[j]):
                    j += 1
                    if j >= i + 10: # if can not find the first line of force in 10 lines, then stop
                        print("Warning: can not find the first line of force")
                        nostress = True
                        break
                if nostress:
                    break
                stress = []
                while value_pattern.match(self.LOG[j]):
                    stress += [float(ii) for ii in self.LOG[j].split()[:3]]
                    j += 1
                if stress: stresses.append(stress)
        
        virials = []
        pressures = []
        for stress in stresses:
            if self["volume"] != None:
                virial = [i * self["volume"] * comm.KBAR2EVPERANGSTROM3 for i in stress]
                virials.append(virial)
            pressures.append((stress[0] + stress[4] + stress[8]) / 3.0)
            
        if stresses:
            self['stress'] = stresses[-1]
            self["stresses"] = stresses
            self["pressure"] = pressures[-1]
            self["pressures"] = pressures
        else:
            self["stress"] = None
            self["stresses"] = None
            self["pressure"] = None
            self["pressures"] = None

        if virials:
            self['virial'] = virials[-1]
            self["virials"] = virials
        else:
            self['virial'] = None
            self["virials"] = None
        
    @ResultAbacus.register(largest_gradient="list, the largest gradient of each ION step. Unit in eV/Angstrom",
                           largest_gradient_stress="list, the largest stress of each ION step. Unit in kbar")
    def GetLargestGradientFromLog(self):
        lg, lg_stress = None, None
        force_lines, force_conflict = _stdout_lines(
            self, lambda lines: _gradient_component(lines, 0),
            "largest force gradient",
        )
        stress_lines, stress_conflict = _stdout_lines(
            self, lambda lines: _gradient_component(lines, 1),
            "largest stress gradient",
        )
        if not force_conflict:
            parsed = _gradient_markers(force_lines)
            if parsed is not _CONFLICT:
                lg = parsed[0]
        if not stress_conflict:
            parsed = _gradient_markers(stress_lines)
            if parsed is not _CONFLICT:
                lg_stress = parsed[1]
        
        if lg is None and not force_conflict:
            forces = self["forces"]
            if forces is not None:
                mobility = _mobility_components(self)
                values = []
                for force in forces:
                    selected = []
                    for atom_index in range(len(force) // 3):
                        atom_mobility = mobility[atom_index] if atom_index < len(mobility) else None
                        for component in range(3):
                            if atom_mobility is None or atom_mobility[component]:
                                selected.append(abs(force[3 * atom_index + component]))
                    values.append(max(selected) if selected else None)
                if any(value is not None for value in values):
                    lg = values
        
        if lg_stress is None and not stress_conflict:
            stresses = self["stresses"]
            if stresses is not None:
                lg_stress = [max([abs(i) for i in j]) for j in stresses]

        self['largest_gradient'] = lg
        self['largest_gradient_stress'] = lg_stress
        return lg[-1] if lg else None

    @ResultAbacus.register(k_coord="list, the direct k point coordinates in the BZ",)
    def GetKCoordFromLog(self):
        coord = []
        nkstot = self["nkstot"]
        for i, line in enumerate(self.LOG):
            if "K-POINTS DIRECT COORDINATES" in line:
                for j in range(i+2, i+2+nkstot):
                    coord.append([float(k) for k in self.LOG[j].split()[1:4]])
                break
        if len(coord) > 0:
            self['k_coord'] = coord
        else:
            self['k_coord'] = None
    
    @ResultAbacus.register(band = "Band of system. Dimension is [nspin,nk,nband].",
                           band_weight = "Band weight of system. Dimension is [nspin,nk,nband]."
                           )
    def GetBand(self): 
        band_file = os.path.join(self.PATH,f"OUT.{self.SUFFIX}/eig.txt") # in new version the band info is in eig.txt
        if os.path.isfile(band_file):
            band = []
            weight = []
            with open(band_file) as f: lines = f.readlines()
            for line in lines[2:]:
                if line.strip() == "":
                    continue
                elif line.startswith(" spin="):
                    spin_idx = int(line[6]) # start with 1
                    if spin_idx > len(band):
                        band.append([])
                        weight.append([])
                    band[-1].append([])
                    weight[-1].append([])
                else:
                    sline = line.split()
                    if len(sline) == 3:
                        band[-1][-1].append(float(sline[1]))
                        weight[-1][-1].append(float(sline[2]))
            self['band'] = band
            self['band_weight'] = weight
        else:
            nband = self['nbands']  
            if self["ibzk"] != None:    
                nk = self['ibzk']
            elif self["nkstot"] != None:
                nk = self["nkstot"]
            else:
                nk = None

            if nband == None or nk == None:
                print("no nbands or ibzk, and skip the catch of band info")
                return

            band = None
            band_weight = None
            for i,line in enumerate(self.LOG):
                if 'STATE ENERGY(eV) AND OCCUPATIONS' in line:
                    nspin = int(line.split()[-1])
                    if nspin == 4: nspin=1  # for nspin4, only total band is output
                    band = []
                    band_weight = []
                    for ispin in range(nspin):
                        band.append([])
                        band_weight.append([])
                        for k in range(nk):
                            band[-1].append([])
                            band_weight[-1].append([])
                            for m in range(nband):
                                ni = ((nband+2)*nk + 1) * ispin + (nband+2)*k + nspin + m + 1 + i
                                eband = float(self.LOG[ni].split()[1])
                                wband = float(self.LOG[ni].split()[2])
                                band[-1][-1].append(eband)
                                band_weight[-1][-1].append(wband)
                    break
            self['band'] = band
            self['band_weight'] = band_weight
    
    @ResultAbacus.register(band_plot="Plot the band structure. Return the file name of the plot.")
    def PlotBandFromLog(self):  
        band = self['band']  
        efermi = self['efermi']
        if band == None:
            print("no band, and skip the plot of band")
            self['band_plot'] = None
            return
        band_plot = os.path.join(self.PATH,"band.png")
        comm.plot_band(band, band_plot, efermi)
        self['band_plot'] = band_plot     
    
    @ResultAbacus.register(e_bandgap= "The band gap outputted in running_xxx.log, unit in eV")
    def GetEBandGapFromLog(self):
        ebandgap = None 
        for line in self.LOG[::-1]:
            if "E_bandgap" in line:
                ebandgap = float(line.split()[-1])
                break
        self['e_bandgap'] = ebandgap
    
    @ResultAbacus.register(band_gap = "band gap of the system")
    def GetBandGapFromLog(self):
        def ErrorReturn(strinfo):
            print(f"WARNING: {strinfo}, skip the catch of band gap info")
            self['band_gap'] = None
            return
        
        band = self['band']
        band_weight = self['band_weight']
        if band == None or band_weight == None:
            ErrorReturn("no band")
            return
        
        efermi = self['efermi']
        if efermi == None:
            ErrorReturn("no efermi")
            return
        
        self['band_gap'] = comm.cal_band_gap(band,efermi)
    '''
        
    @ResultAbacus.register(band_gap = "band gap of the system")
    def GetBandGapFromLog(self):
        def ErrorReturn(strinfo):
            print("WARNING: %s, skip the catch of band gap info" % strinfo)
            self['band_gap'] = None
            return
        
        nband = self['nbands']  
        if self["ibzk"] != None:    
            nk = self['ibzk']
        elif self["nkstot"] != None:
            nk = self["nkstot"]
        else:
            nk = None
            
        if nband == None or nk == None:
            ErrorReturn("no nbands or ibzk")
            return
        nelec = self['nelec']
        if nelec == None:
            ErrorReturn("no nelec")
            return
        occu_band = int(nelec/2)
        if occu_band == 0:
            ErrorReturn("no occu_band")
            return
        if occu_band >= nband:
            ErrorReturn("occu_band >= nband")
            return
                
        band_gap = None
        #print("nelec:",nelec,"occu_band:",occu_band,"nband:",nband,"nk:",nk)
        for i,line in enumerate(self.LOG):
            if 'STATE ENERGY(eV) AND OCCUPATIONS' in line:
                nspin = int(line.split()[-1])
                if nspin not in [1,2]:
                    ErrorReturn("NOT SUPPORT FOR NSPIN=%d now" % nspin)
                    return

                totalcb = None
                totalvb = None
                for ispin in range(nspin):
                    cb = None
                    vb = None
                    for k in range(nk):
                        eband1 = float(self.LOG[((nband+2)*nk + 1) * ispin + (nband+2)*k + nspin + occu_band + i].split()[1])  # the highest occupied band
                        eband2 = float(self.LOG[((nband+2)*nk + 1) * ispin + (nband+2)*k + nspin + occu_band + 1 + i].split()[1])  # the lowest unoccupied band
                        if cb == None or eband1 > cb:
                            cb = eband1
                        if vb == None or eband2 < vb:
                            vb = eband2
                        #print("k:",k,"eband1:",eband1,"eband2:",eband2,"cb:",cb,"vb:",vb)
                    if totalcb == None or (cb != None and totalcb < cb): totalcb = cb
                    if totalvb == None or (vb != None and totalvb > vb): totalvb = vb
                    #print("ispin:",ispin,"totalcb:",totalcb,"totalvb:",totalvb)
                if totalcb == None or totalvb == None:
                    band_gap = None
                else:
                    band_gap = totalvb-totalcb
                    if band_gap < 0: band_gap = 0
                break
                
        self['band_gap'] = band_gap
    '''
    
    @ResultAbacus.register(total_time="the total time of the job",
                           stress_time="the time to do the calculation of stress",
                           force_time = "the time to do the calculation of force",
                           scf_time = "the time to do SCF",
                           scf_time_each_step = "list, the time of each step of SCF",
                           step1_time = "the time of 1st SCF step",
                           scf_steps = "the steps of SCF")
    def GetTimeFromOutput(self):
        # first, check if self.time is already exist
        if self.TIME:
            total_time = self.GetTime("total",None)[0]
            # if PW basis, stress time = Stress_PW/cal_stress, force_time = cal_force_nl
            # if lcao basis, stress time = getForceStress, force_time = None
            stress_time_pw = self.GetTime("Stress_PW","cal_stress")[0]
            stress_time_lcao = self.GetTime("Force_Stress_LCAO","getForceStress")[0]
            force_time_pw = self.GetTime("Forces","cal_force_nl")[0]
            force_time_lcao = None
            if stress_time_pw != None:
                stress_time = stress_time_pw
            else:
                stress_time = stress_time_lcao
                
            if force_time_pw != None:
                force_time = force_time_pw
            else:
                force_time = force_time_lcao
        else:
            stress_time = None
            force_time = None
            total_time = None
            for i,line in enumerate(self.OUTPUT):
                if line[23:28] == 'total' or (len(line.split()) == 5 and line.split()[0] == 'total'):  # old version or new version
                    total_time = float(line.split()[1])
                elif line[23:33] == 'cal_stress' or (len(line.split()) == 6 and line.split()[0] == 'cal_stress'): # old version or new version
                    stress_time = float(line.split()[-5])
                elif line[23:35] == 'cal_force_nl' or (len(line.split()) == 6 and line.split()[0] == 'cal_force_nl'):
                    force_time = float(line.split()[-5])
                elif line[23:37] == 'getForceStress' or (len(line.split()) == 6 and line.split()[0] == 'getForceStress'):
                    stress_time = float(line.split()[-5])
                    force_time = None
                    
        self["total_time"] = total_time
        self['stress_time'] = stress_time
        self['force_time'] = force_time

        scftime = []
        for i,line in enumerate(self.OUTPUT):
            if line[1:5] == 'ITER':
                for j in range(i+1,len(self.OUTPUT)):
                    if self.OUTPUT[j][1:3] in KS_SOLVER_LIST or (self.OUTPUT[j].startswith("1") and self.OUTPUT[j][2:4] in KS_SOLVER_LIST): 
                        scftime.append(float(self.OUTPUT[j].split()[-1]))
                    elif self.OUTPUT[j].startswith(" -----------------------------------"):
                        break
        if len(scftime) > 0:
            self['scf_time'] = np.array(scftime).sum()
            self['step1_time'] = scftime[0]
            self['scf_steps'] = len(scftime)
            self['scf_time_each_step'] = scftime

    @ResultAbacus.register(
                           atom_mag = "list, the magnization of each atom. Only the last ION step. Calculated by atomic orbital projection.",
                           atom_mags= "list atom mag of each ION step. Calculated by atomic orbital projection.",
                         )
    def GetAtomMagOrbital(self):
        self["atom_mags"] = self["ds_mags"]
        self["atom_mag"] = self["ds_mag"]
    
    @ResultAbacus.register(atom_orb_mag = "list of atomic orbital magnization. Only the last ION step. Calculated by atomic orbital projection.",
                            atom_orb_mags = "list of atomic orbital magnization of each ION step. Calculated by atomic orbital projection.",
                            atom_elec = "list of atomic electron. Electron calculated by atomic orbital projection.",
                            atom_elecs="list of atomic electron of each ION step. Electron calculated by atomic orbital projection.",
                            atom_orb_elec="list of atomic orbital electron. Electron calculated by atomic orbital projection.",
                            atom_orb_elecs="list of atom_orb_elec of each ION step. Electron calculated by atomic orbital projection.",
                          )
    def GetAtomMagOrbital(self):
        """Get electrion and mag from orbital charge analysis.
        """
        
        '''
-------------------------------------------------------------------------------------------
Orbital Charge Analysis      Charge         Mag(x)         Mag(y)         Mag(z)
-------------------------------------------------------------------------------------------
Fe1
                   s         1.0799        -0.0000         0.0000         0.0034
                   p         5.9969        -0.0000         0.0000         0.0004
                   d         6.5446        -0.0039         0.0013         3.0690
                 Sum        13.6214        -0.0039         0.0013         3.0729
Fe2
                   s         1.0827         0.0000        -0.0000         0.0040
                   p         5.9969         0.0000        -0.0000         0.0004
                   d         6.6497         0.0025        -0.0002         2.9733
                 Sum        13.7294         0.0025        -0.0002         2.9776
-------------------------------------------------------------------------------------------        
        '''
        
        atom_mags = []
        atom_orb_mags = []
        atom_elecs = []
        atom_orb_elecs = []
        for i, line in enumerate(self.LOG):
            if "Orbital Charge Analysis      Charge" in line:
                atom_mag = []
                atom_orb_mag = []
                atom_elec = []
                atom_orb_elec = []
                j = i + 2
                while "-------" not in self.LOG[j]:
                    sline = self.LOG[j].split()
                    if len(sline) == 1:
                        atom_orb_mag.append([])
                        atom_orb_elec.append([])
                    else:
                        if "Sum" in sline:
                            atom_elec.append(float(sline[1]))
                            atom_mag.append([float(ii) for ii in sline[2:]])
                        else:
                            atom_orb_mag[-1].append(float(sline[1]))
                            atom_orb_elec[-1].append([float(ii) for ii in sline[2:]])
                    j += 1
                atom_mags.append(atom_mag)
                atom_orb_mags.append(atom_orb_mag)
                atom_elecs.append(atom_elec)
                atom_orb_elecs.append(atom_orb_elec)
        
        if len(atom_orb_mags) == 0:
            self["atom_orb_mags"] = None
            self["atom_orb_mag"] = None
            self["atom_elecs"] = None
            self["atom_elec"] = None
            self["atom_orb_elecs"] = None
            self["atom_orb_elec"] = None
        else:
            self["atom_orb_mags"] = atom_orb_mags
            self["atom_orb_mag"] = atom_orb_mags[-1]
            self["atom_elecs"] = atom_elecs
            self["atom_elec"] = atom_elecs[-1]
            self["atom_orb_elecs"] = atom_orb_elecs
            self["atom_orb_elec"] = atom_orb_elecs[-1]
    
    @ResultAbacus.register(atom_mag_mul = "list of mulliken magnization of each atom. Only the last ION step.",
                           atom_mags_mul="list of atom_mag_mul of each ION step.",
                           atom_elec_mul="list of mulliken atomic electron.",
                           atom_orb_elec_mul="list of mulliken atomic orbital electron.",
                           atom_elecs_mul="list of mulliken atomic electron of each ION step.",
                           atom_orb_elecs_mul="list of mulliken atomic orbital electron of each ION step",
                           )
    def GetAtomMagMul(self):
        mullikenf = os.path.join(os.path.split(self.LOGf)[0],"mulliken.txt")
        if not os.path.isfile(mullikenf):
            self['atom_mag_mul'] = None
            self["atom_mags_mul"] = None
            self["atom_elec"] = None
            return
        
        atom_mag, atom_elec, _ = comm.get_mulliken(mullikenf)
        
        if len(atom_mag) == 0:
            self['atom_mags_mul'] = None
            self["atom_mag_mul"] = None
        else:
            self['atom_mags_mul'] = atom_mag
            self["atom_mag_mul"] = atom_mag[-1]
        
        if not atom_elec:
            self["atom_elec_mul"] = None
            self["atom_orb_elec_mul"] = None
            self["atom_elecs_mul"] = None
            self["atom_orb_elecs_mul"] = None
        else:
            self["atom_orb_elecs_mul"] = atom_elec # atom orb elecs of each ION step
            self["atom_orb_elec_mul"] = atom_elec[-1] # atom orb elec of last ION step
            atom_elecs = []
            for atom_orb_elec in self["atom_orb_elecs_mul"]:
                atom_elec = []
                for i in atom_orb_elec:
                    atom_elec.append(sum([sum(j) for j in i]))
                atom_elecs.append(atom_elec)
            self["atom_elecs_mul"] = atom_elecs
            self["atom_elec_mul"] = atom_elecs[-1]
            
    '''
    repeat to charge, not used anymore
    @ResultAbacus.register(atom_mag_uorb="list of a dict, the magnization of each atom U orbitals. Only the last SCF step.",
                           atom_elec_uorb = "list of a dict with keys are atom index, atom label, and electron of orbital adding U.")
    def GetAtomMag(self):
        atom_mag_u = None
        atom_elec_u = None
        if self.LOG:
            u_block = []
            start_line = end_line = None
            for i in  range(len(self.LOG)):
                i = -1*i - 1
                if "//=========================L(S)DA+U===========================//" in self.LOG[i]:
                    start_line = i + 1
                    break
                elif "//=======================================================//" in self.LOG[i]:
                    end_line = i
            print(start_line,end_line)
            if None not in [start_line, end_line]:
                u_block = self.LOG[start_line:end_line]
            else:
                print("Can not find the L(S)DA+U block")
                self['atom_mag_uorb'] = None
                self["atom_elec_uorb"] = None
                return
            
            labels = self["label"]
            atom_mag_u = []
            atom_elec_u = []
            for i,line in enumerate(u_block):
                if "atoms" in line:
                    atom_index = int(line.split()[-1])
                    atom_label = None if not labels else labels[atom_index]
                    atom_mag_u.append({"index":atom_index,"label":atom_label,"mag":None})
                    atom_elec_u.append({"index":atom_index,"label":atom_label,"elec":None})
                elif "eigenvalues" in line:
                    if atom_elec_u[-1]["elec"] == None:
                        atom_elec_u[-1]["elec"] = []
                    atom_elec_u[-1]["elec"].append(float(u_block[i+1].split()[-1]))
                elif "atomic mag:" in line:
                    atom_mag_u[-1]["mag"] = float(line.split()[-1])
        self['atom_mag_uorb'] = atom_mag_u
        self['atom_elec_uorb'] = atom_elec_u
    '''
    
    @ResultAbacus.register(drho="[], drho of each scf step",
                           drho_last="drho of the last scf step")
    def GetDrho(self):
        drho = []
        for line in self.LOG:
            if "Density error is" in line:
                drho.append(float(line.split()[-1]))
        
        if len(drho) == 0:
            self['drho'] = None
        else:
            self['drho'] = drho
            self["drho_last"] = drho[-1]
    
    @ResultAbacus.register(denergy="[], denergy of each scf step",
                           denergy_last="denergy of the last scf step")
    def GetDenergy(self):
        denergy = None
        if self.OUTPUT:
            for i,line in enumerate(self.OUTPUT):
                if "ITER" in line and "ETOT/eV" in line and "EDIFF/eV" in line and "DRHO" in line and "TIME/s" in line:
                    denergy = []
                    ncol = len(line.split())
                    ediff_idx = line.split().index("EDIFF/eV")
                    j = i + 1
                    while j < len(self.OUTPUT):
                        if "----------------------------" in self.OUTPUT[j]:
                            break
                        if self.OUTPUT[j][1:3] in KS_SOLVER_LIST and len(self.OUTPUT[j].split()) == ncol:
                            denergy.append(float(self.OUTPUT[j].split()[ediff_idx]))
                        j += 1

        self["denergy"] = denergy
        if denergy != None:
            self["denergy_last"] = denergy[-1]
        else:
            self["denergy_last"] = None
    
    @ResultAbacus.register(denergy_womix="[], denergy (calculated by rho without mixed) of each scf step",
                           denergy_womix_last="float, denergy (calculated by rho without mixed) of last scf step")
    def GetDenergyWOMIX(self):
        des = []
        if self.LOG:
            for line in self.LOG:
                if "DeltaE_womix" in line:
                    des.append(float(line.split()[-2]))
        if len(des) == 0:
            self["denergy_womix"]    = None
            self["denergy_womix_last"] = None
        else:
            des[0] = 0
            self["denergy_womix"]    = des
            self["denergy_womix_last"] = des[-1]


    @ResultAbacus.register(lattice_constant="a list of six float which is a/b/c,alpha,beta,gamma of cell. If has more than one ION step, will output the last one.",
                           lattice_constants="a list of list of six float which is a/b/c,alpha,beta,gamma of cell",
                           cell = "[[],[],[]], two-dimension list, unit in Angstrom. If is relax or md, will output the last one.",
                           cells = "a list of [[],[],[]], which is a two-dimension list of cell vector, unit in Angstrom.",
                           cell_init = "[[],[],[]], two-dimension list, unit in Angstrom. The initial cell",
                           coordinate = "[[],..], two dimension list, is a cartesian type, unit in Angstrom. If is relax or md, will output the last one",
                           coordinates = "[[[], ...], ...], three dimension list, is a cartesian type, unit in Angstrom. The coordinate of each ION step If is relax or md",
                           coordinate_init = "[[],..], two dimension list, is a cartesian type, unit in Angstrom. The initial coordinate",
                           element = "list[], a list of the element name of all atoms",
                           label = "list[], a list of atom label of all atoms",
                           element_list = "same as element",
                           atomlabel_list = "same as label")
    def GetCell(self): 
        lc = 1   
        for line in self.LOG:
            if "lattice constant (Angstrom)" in line:
                lc = float(line.split()[-1])
                break
        
        cells = []
        for i in range(len(self.LOG)):
            line = self.LOG[i]
            if "Lattice vectors: (Cartesian coordinate: in unit of a_0)" in line:
                cell = []
                for k in range(1, 4):
                    cell.append([float(x) * lc for x in self.LOG[i + k].split()[0:3]])
                cells.append(cell)
                    
        if cells:
            self['cells'] = cells
            self['cell'] = cells[-1]
            # calculate a/b/c,alpha,beta,gamma
            lattice_constants = []
            for cell in cells:
                a = np.linalg.norm(cell[0])
                b = np.linalg.norm(cell[1])
                c = np.linalg.norm(cell[2])
                alpha = np.arccos(np.dot(cell[1],cell[2])/(b*c))*180/np.pi
                beta = np.arccos(np.dot(cell[0],cell[2])/(a*c))*180/np.pi
                gamma = np.arccos(np.dot(cell[0],cell[1])/(a*b))*180/np.pi
                lattice_constants.append([a,b,c,alpha,beta,gamma])
            self['lattice_constants'] = lattice_constants
            self['lattice_constant'] = lattice_constants[-1]
        else:
            self["cells"] = None
            self["cell"] = None
            self['lattice_constants'] = None
            self['lattice_constant'] = None
        
        cell_init = None
        for i in range(len(self.LOG)): 
            iline = i 
            line = self.LOG[iline]
            if "Lattice vectors: (Cartesian coordinate: in unit of a_0)" in line:
                cell_init = []
                for k in range(1, 4):
                    cell_init.append([float(x) * lc for x in self.LOG[iline + k].split()[0:3]])
                break
        self['cell_init'] = cell_init
        
        coordinate = None
        coordinate_init = None
        natom = self['natom']
        if cell != None and natom != None:
            try:
                coordinates = []
                for i in range(len(self.LOG)):
                    iline = i
                    line = self.LOG[iline]
                    if len(line.split()) >= 2 and line.split()[1] == "COORDINATES":  
                        coordinate = []
                        if line.split()[0] == "DIRECT":
                            for k in range(2, 2 + natom):
                                coordinate.append([float(x) for x in self.LOG[iline + k].split()[1:4]])
                            coordinate = np.array(coordinate).dot(np.array(cell)).tolist()
                            coordinates.append(coordinate)
                        elif line.split()[0] == "CARTESIAN":
                            for k in range(2, 2 + natom):
                                coordinate.append([float(x) for x in self.LOG[iline + k].split()[1:4]])
                            coordinates.append(coordinate)
                        else:
                            print("Unrecongnized coordinate type: %s" % (line))   
                
                if len(coordinates) > 0:
                    self['coordinates'] = coordinates
                    self['coordinate'] = coordinates[-1]
                    self['coordinate_init'] = coordinates[0]
                else:
                    self['coordinates'] = None
                    self['coordinate'] = None
                    self['coordinate_init'] = None
            except:
                traceback.print_exc()
                self['coordinates'] = None
                self['coordinate'] = None
                self['coordinate_init'] = None

        else:
            print("No cell or natom, skip the catch of coordinate info")
            self['coordinate'] = None
            self['coordinate_init'] = None
              
        element_list = None
        atomlabel_list = None
        for line in self.LOG:
            if "atom label =" in line:
                label = line.split()[-1]
                element = label
                while element[-1].isdigit():
                    element = element[:-1]
            elif "number of atom for this type =" in line:
                if element_list == None: element_list = []
                if atomlabel_list == None: atomlabel_list = []
                natom = int(line.split()[-1])
                atomlabel_list += natom * [label]
                element_list += natom * [element]
                if len(atomlabel_list) >= self['natom']:
                    break
        self["element"] = element_list
        self["label"] = atomlabel_list
        self['element_list'] = element_list
        self['atomlabel_list'] = atomlabel_list
    
    @ResultAbacus.register(dos="dict, where keys are 'energy' and 'total', and value of 'total' is a dict of (nenergy,spin)",)
    def GetDOS(self):
        import glob
        dosfiles = glob.glob(os.path.join(self.PATH,f"OUT.{self.SUFFIX}","DOS*_smearing.dat"))
        print("DOS file: ", dosfiles)
        dos_datas = []
        for dosfile in dosfiles:
            dos_data = np.loadtxt(dosfile)
            energy = dos_data[:, 0].tolist()
            dos_datas.append(dos_data[:, 1])

        if len(dos_datas) == 1: # nspin = 1 or 4
            dos_datas = dos_datas[0].T.reshape(-1, 1).tolist()
        else:
            dos_datas = np.array(dos_datas).T.tolist()
        
        dos = {'energy': energy, 'data': dos_datas}

        self['dos'] = dos

    @ResultAbacus.register(pdos="dict, where keys are 'energy' and 'orbitals', and value of 'orbitals' is a dict of (index,species,l,m,z,data)",
                           )
    def GetPDOS(self): 
        pdos_file = os.path.join(self.PATH,f"OUT.{self.SUFFIX}","PDOS")
        print("PDOS file path:",pdos_file)
        pdos = None
        if os.path.isfile(pdos_file):
            import xml.etree.ElementTree as ET
            from io import StringIO
            tree = ET.parse(pdos_file)
            root = tree.getroot()
            energy = [float(i) for i in root.find('energy_values').text.split()]

            all_orbitals = []
            for iorb in root.findall('orbital'):
                pdos_data = np.loadtxt(StringIO(iorb.find('data').text))
                if len(pdos_data.shape) == 1:
                    pdos_data = pdos_data.reshape(-1, 1)
                pdos_data = pdos_data.tolist()
                
                orbital_info = {
                    "index": int(iorb.get('index')),
                    "atom_index": int(iorb.get('atom_index')),
                    "species": iorb.get('species'),
                    "l": int(iorb.get('l')),
                    "m": int(iorb.get('m')),
                    "z": int(iorb.get('z')),
                    "data": pdos_data
                }
                all_orbitals.append(orbital_info)
            pdos = {"energy":energy,"orbitals":all_orbitals}
        self["pdos"] = pdos

class AbacusRelax(ResultAbacus):
    @ResultAbacus.register(relax_converge="if the relax is converged")
    def GetRelaxConverge(self):
        converge = None
        evidence, status_conflict = _stdout_lines(self, _relax_status, "relaxation status")
        if not status_conflict:
            converge = _relax_status(evidence)

        # In some versions ABACUS does not print an explicit statement.  Use
        # only evidence that is actually present; a missing stdout must remain
        # unknown instead of being reported as a failed relaxation.
        if converge is None and not status_conflict:
            evidence, gradient_conflict = _stdout_lines(
                self,
                lambda lines: _gradient_component(lines, 0),
                "largest force gradient",
            )
            if gradient_conflict:
                self["relax_converge"] = None
                return None
            lg_force = self["largest_gradient"]
            lg_stress = self["largest_gradient_stress"]
            params = self["INPUT"]
            job_type = params.get("calculation", "scf")
            force_thr = params.get("force_thr_ev")
            if force_thr is None and params.get("force_thr") is not None:
                force_thr = float(params["force_thr"]) * 25.711396132
            if isinstance(lg_force, list) and lg_force:
                latest_force = lg_force[-1]
            else:
                latest_force = lg_force
            force_ok = (
                latest_force is not None and force_thr is not None and
                latest_force <= float(force_thr)
            )
            if job_type == "relax" and latest_force is not None and force_thr is not None:
                relax_new = params.get("relax_new", 1)
                is_new = str(relax_new).strip().lower() not in ("0", "false", "no", "off")
                if is_new:
                    converge = force_ok
                else:
                    # The legacy relaxation path has two required pieces of
                    # evidence: force and electronic energy difference.
                    energy_lines, energy_conflict = _stdout_lines(
                        self, _energy_difference, "energy difference"
                    )
                    energy_difference = None if energy_conflict else _energy_difference(energy_lines)
                    energy_thr = params.get("scf_ene_thr")
                    if energy_thr is None:
                        energy_thr = params.get("energy_thr", params.get("relax_energy_thr"))
                    if energy_difference is not None and energy_thr is not None and float(energy_thr) >= 0:
                        converge = force_ok and abs(energy_difference) <= float(energy_thr)
            elif job_type == "cell-relax" and force_ok:
                if isinstance(lg_stress, list) and lg_stress:
                    stress_thr = params.get("stress_thr")
                    converge = stress_thr is not None and lg_stress[-1] <= float(stress_thr)
                else:
                    converge = None

        self["relax_converge"] = converge
        return converge

    @ResultAbacus.register(relax_steps= "the total ION steps")
    def GetRelaxSteps(self):
        #need read self.LOG
        if self.LOG:
            for i in range(len(self.LOG)):
                line = self.LOG[-i-1]
                if "ALGORITHM --------------- ION=" in line:
                    index_ben = line.index("ION=") + 4
                    index_end = line.index("ELEC")
                    self["relax_steps"] = int(line[index_ben:index_end])
                    return
                elif "STEP OF RELAXATION :" in line:
                    self["relax_steps"] = int(line.split()[-1])
                    return
                elif " STEP OF ION RELAXATION : " in line:
                    self["relax_steps"] = int(line.split()[-1])
                    return
        self["relax_steps"] = None


class AbacusDeltaSpin(ResultAbacus):
    
    @ResultAbacus.register(ds_lambda_step="a list of DeltaSpin converge step in each SCF step",
                           ds_lambda_rms="a list of DeltaSpin RMS in each SCF step",
                           ds_time="a list of the total time of inner loop in deltaspin for each scf step.")
    def GetDSLog(self):
        """Get metrcis of DeltaSpin from the log file.
        """
        
        '''
Step (Outer -- Inner) =  16 -- 1           RMS = 1.057e-07
Step (Outer -- Inner) =  16 -- 2           RMS = 1.355e-07
Step (Outer -- Inner) =  16 -- 3           RMS = 1.176e-07
Step (Outer -- Inner) =  16 -- 4           RMS = 9.760e-08

        '''
        lambda_step = None 
        lambda_rms = None
        ds_time = None
        if self.OUTPUT:
            scf_step = []
            lambda_step = []
            lambda_rms = []
            for idx, i in enumerate(self.OUTPUT):
                if "Step (Outer -- Inner) =" in i:
                    ss = int(i.split()[5])
                    lambdas = int(i.split()[7])
                    rms = float(i.split()[10])
                    if ss not in scf_step:
                        scf_step.append(ss)
                        lambda_step.append(lambdas)
                        lambda_rms.append(rms)
                    else:
                        lambda_step[-1] = lambdas
                        lambda_rms[-1] = rms
                elif "Meet convergence criterion" in i and "Total TIME(s) =" in i:
                    if ds_time is None:
                        ds_time = []
                    ds_time.append(float(i.split()[-1]))

        self["ds_lambda_step"] = lambda_step
        self["ds_lambda_rms"] = lambda_rms   
        self["ds_time"] = ds_time
        
    @ResultAbacus.register(ds_mag="a list of list, each element list is for each atom. Unit in uB",
                           ds_mags="list of ds_mag of each ION step. Unit in uB",
                           ds_mag_force="a list of list, each element list is for each atom. Unit in eV/uB",
                           ds_mag_forces="list of ds_mag_force of each ION step. Unit in eV/uB")
    def GetDSOutput(self):
        '''
===============================================================================
 DA13     1.98e+00  -7.38e-10   3.44e+00   4.88e+00  -6.81997451e+03   1.78754645e-06   2.5764e-09 108.32
===============================================================================
Inner optimization for lambda begins ...
Covergence criterion for the iteration: 1e-08
initial lambda (eV/uB):
ATOM      1         0.1482429405         0.0000000026         0.0156735253
ATOM      2        -0.0605498198         0.0000000025         0.1362180540
initial spin (uB):
ATOM      1         0.0000367975        -0.0000000063         2.4319892847
ATOM      2         2.1061545741        -0.0000000061         1.2160301143
target spin (uB):
ATOM      1         0.0000000000         0.0000000000         2.4320000000
ATOM      2         2.1061737820         0.0000000000         1.2160000000
Step (Outer -- Inner) =  13 -- 1           RMS = 3.70452e-05     TIME(s) = 24.4197
Step (Outer -- Inner) =  13 -- 2           RMS = 3.15459e-06     TIME(s) = 16.202
Step (Outer -- Inner) =  13 -- 3           RMS = 8.99985e-07     TIME(s) = 16.2116
Step (Outer -- Inner) =  13 -- 4           RMS = 1.42306e-07     TIME(s) = 16.2354
Step (Outer -- Inner) =  13 -- 5           RMS = 3.15143e-08     TIME(s) = 16.2353
Meet convergence criterion ( < 3.70452e-08 ), exit.       Total TIME(s) = 89.304
after-optimization spin (uB): (print in the inner loop):
ATOM      1        -0.0000000053        -0.0000000001         2.4319999813
ATOM      2         2.1061737840        -0.0000000001         1.2159999599
after-optimization lambda (eV/uB): (print in the inner loop):
ATOM      1         0.1482675982        -0.0000000006         0.0156621842
ATOM      2        -0.0605667962        -0.0000000007         0.1362365586
Inner optimization for lambda ends.  
...
        '''
        ds_mag1 = []
        ds_mag2 = []
        mag_forces = []

        if self.LOG:
            natom = self["natom"]
            for idx, i in enumerate(self.LOG):
                if "Total Magnetism (uB)" in i:
                    ds_mag = []
                    for j in range(natom):
                        if len(self.LOG[idx+j+2].split()) in [2,4]:
                            ds_mag.append([float(ii) for ii in self.LOG[idx+j+2].split()[1:]])
                    ds_mag1.append(ds_mag)
                elif "Orbital Charge Analysis      Charge" in i:
                    ds_mag = []
                    j = idx +2
                    while "-------" not in self.LOG[j]:
                        if "Sum" in self.LOG[j]:
                            ds_mag.append([float(ii) for ii in self.LOG[j].split()[2:]])   
                        j += 1 
                    ds_mag2.append(ds_mag)    
                elif "Magnetic force (eV/uB)" in i:
                    mag_force = []
                    for j in range(natom):
                        if len(self.LOG[idx+j+2].split()) in [2,4]:
                            mag_force.append([float(ii) for ii in self.LOG[idx+j+2].split()[1:]])
                    mag_forces.append(mag_force)

        if len(ds_mag1) == 0 and len(ds_mag2) == 0:
            self["ds_mag"] = None
            self["ds_mags"] = None
        elif len(ds_mag1) > 0:
            self["ds_mag"] = ds_mag1[-1]
            self["ds_mags"] = ds_mag1
        else:
            self["ds_mag"] = ds_mag2[-1]
            self["ds_mags"] = ds_mag2
        if len(mag_forces) == 0:
            self["ds_mag_force"] = None
            self["ds_mag_forces"] = None
        else:
            self["ds_mag_forces"] = mag_forces
            self["ds_mag_force"] = mag_forces[-1]

        
class AbacusMemory(ResultAbacus):
    
    @ResultAbacus.register(mem_vkb="the memory of VNL::vkb, unit it MB",
                           mem_psipw="the memory of PsiPW, unit it MB",)
    def GetMemory(self):
        if self.LOG:
            for line in self.LOG:
                if "Warning_Memory_Consuming allocated:  VNL::vkb" in line:
                    self["mem_vkb"] = float(line.split()[-2])
                elif "Warning_Memory_Consuming allocated:  Psi_PW" in line:
                    self["mem_psipw"] = float(line.split()[-2])
                    

         
