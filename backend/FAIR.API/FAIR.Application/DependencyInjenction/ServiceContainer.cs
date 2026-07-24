using FAIR.API.Interfaces.Report;
using FAIR.Application.Mapping;
using FAIR.Application.Services.Implementations.Authentication;
using FAIR.Application.Services.Implementations.Profile;
using FAIR.Application.Services.Implementations.Reports;
using FAIR.Application.Services.Interfaces;
using FAIR.Application.Validations;
using FAIR.Application.Validations.Authentication;
using FluentValidation;
using FluentValidation.AspNetCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace FAIR.Application.DependencyInjenction
{
    public static class ServiceContainer
    {
        public static IServiceCollection AddApplicationService
            (this IServiceCollection services )
        {
            services.AddFluentValidationAutoValidation();
            services.AddValidatorsFromAssemblyContaining<CreateUserValidator>();
            services.AddScoped<IReportService, ReportService>();
            services.AddScoped<IAuthenticationService, AuthenticationService>();
            services.AddAutoMapper(typeof(MappingConfig));
            services.AddScoped<IValidationService, ValidationService>();
            services.AddScoped<IUserService, UserService>();

            return services;
        }
    }
}
