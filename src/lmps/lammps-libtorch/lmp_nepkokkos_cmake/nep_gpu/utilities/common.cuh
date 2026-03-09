/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.

***
Only the constant definitions related to NEP and ZBL are retained, compared to the GPUMD sourcecode src/utilities/common.cuh
    wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.
***
*/
#pragma once

#define NEIGHMASK 0x1FFFFFFF
constexpr int NUM_ELEMENTS = 103;
constexpr float K_C_SP = 14.399645f;                            // 1/(4*PI*epsilon_0)